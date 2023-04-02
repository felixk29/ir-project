import os
from argparse import ArgumentParser, Namespace
from pprint import pformat
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from attr import asdict
import allrank.models.metrics as allrank_metrics

from sklearn import metrics

from voting_rules import Borda, Copeland

from allrank.click_models.click_utils import click_on_slates
from allrank.config import Config
from torch.utils.data import DataLoader
from allrank.data.dataset_loading import load_libsvm_dataset_role
from allrank.data.dataset_saving import write_to_libsvm_without_masked
from allrank.inference.inference_utils import rank_slates, metrics_on_clicked_slates
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel, load_state_dict_from_file
from allrank.utils.args_utils import split_as_strings
from allrank.utils.command_executor import execute_command
from allrank.utils.config_utils import instantiate_from_recursive_name_args
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import all_equal

from typing import Tuple, Dict, List, Generator

from torch.utils.data.dataloader import DataLoader
from allrank.data.dataset_loading import LibSVMDataset
from allrank.models.model import LTRModel
import allrank.models.losses as losses

MQ_2008_PATH = "/home/enrique/IR/information_retrieval/data/mq2008"
WEB10K_PATH = "/home/enrique/IR/information_retrieval/data/web10k"

PATHS_MQ_2008 = [
    ("/home/enrique/IR/information_retrieval/configs/base_trains/approxmq2008.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/approxmq2008/model.pkl"),
    ("/home/enrique/IR/information_retrieval/configs/base_trains/lambdamq2008.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/lambdamq2008/model.pkl"),
    ("/home/enrique/IR/information_retrieval/configs/base_trains/msemq2008.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/msemq2008/model.pkl"),
    ("/home/enrique/IR/information_retrieval/configs/base_trains/neuralmq2008.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/neuralmq2008/model.pkl"),
    ("/home/enrique/IR/information_retrieval/configs/base_trains/ranknetmq2008.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/ranknetmq2008/model.pkl")
    ]

PATHS_WEB10K = [
    ("/home/enrique/IR/information_retrieval/configs/base_trains/approxweb10k.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/approxweb10k/model.pkl"),
    ("/home/enrique/IR/information_retrieval/configs/base_trains/lambdaweb10k.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/lambdaweb10k/model.pkl"),
    ("/home/enrique/IR/information_retrieval/configs/base_trains/mseweb10k.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/mseweb10k/model.pkl"),
    ("/home/enrique/IR/information_retrieval/configs/base_trains/neuralweb10k.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/neuralweb10k/model.pkl"),
    ("/home/enrique/IR/information_retrieval/configs/base_trains/ranknetweb10k.json",
        "/home/enrique/IR/information_retrieval/runs/base_trains/results/ranknetweb10k/model.pkl")
    ]

ATS = [5,10,30,60]

def parse_args() -> Namespace:
    parser = ArgumentParser("evaluate test data")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    # parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with model config")
    parser.add_argument("--dataset", required=True, type=str, help="Name of the dataset to use (mq2008 or web10k)")
    # parser.add_argument("--input-model-path", required=True, type=str, help="Path to the model to read weights")
    parser.add_argument("--roles", required=True, type=split_as_strings,
                        help="List of comma-separated dataset roles to load and process")

    return parser.parse_args()


def score_slates(datasets: Dict[str, LibSVMDataset], model: LTRModel, config: Config) \
        -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Ranks given datasets according to a given model

    :param datasets: dictionary of role -> dataset that will be ranked
    :param model: a model to use for scoring documents
    :param config: config for DataLoaders
    :return: dictionary of role -> ranked dataset
        every dataset is a Tuple of torch.Tensor - storing X and y in the descending order of the scores.
    """

    dataloaders = {role: __create_data_loader(ds, config) for role, ds in datasets.items()}

    scores = {role: __score_data(dl, model) for role, dl in dataloaders.items()}

    return scores

def __create_data_loader(ds: LibSVMDataset, config: Config) -> DataLoader:
    return DataLoader(ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)

def __score_data(dataloader: DataLoader, model: LTRModel) -> Tuple[torch.Tensor, torch.Tensor]:
    scored_x = []
    scored_y = []
    model.eval()
    device = get_torch_device()
    with torch.no_grad():
        for xb, yb, _ in dataloader:
            X = xb.type(torch.float32).to(device=device)
            y_true = yb.to(device=device)

            input_indices = torch.ones_like(y_true).type(torch.long)
            mask = (y_true == losses.PADDED_Y_VALUE)
            scores = model.score(X, mask, input_indices)

            scores[mask] = float('-inf')
            scored_x.append(X)
            scored_y.append(scores)

    combined_X = torch.cat(scored_x).cpu()
    combined_y = torch.cat(scored_y).cpu()
    return combined_X, combined_y


def run():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Load arguments
    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, "")

    os.makedirs(paths.base_output_path, exist_ok=True)

    create_output_dirs(paths.output_dir)
    logger = init_logger(paths.output_dir)

    logger.info("will save data in {output_dir}".format(output_dir=paths.base_output_path))

    # Select and load dataset
    if args.dataset == "mq2008":
        data_path = MQ_2008_PATH
        model_paths = PATHS_MQ_2008
    elif args.dataset == "web10k":
        data_path = WEB10K_PATH
        model_paths = PATHS_WEB10K

    datasets = {role: load_libsvm_dataset_role(role, data_path, 240) for role in args.roles}

    n_features = [ds.shape[-1] for ds in datasets.values()]
    assert all_equal(n_features), f"Last dimensions of datasets must match but got {n_features}"

    # gpu support
    dev = get_torch_device()
    logger.info("Will use device {}".format(dev.type))    

    test_scores_per_model = {}

    # Predict test scores with each of the models
    for i, (config_path, weights_path) in enumerate(model_paths):
        config = Config.from_json(config_path)
        logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

        # instantiate model
        model = make_model(n_features=n_features[0], **asdict(config.model, recurse=False))
        model.load_state_dict(load_state_dict_from_file(weights_path, dev))
        logger.info(f"loaded model weights from {weights_path}")

        if torch.cuda.device_count() > 1:
            model = CustomDataParallel(model)
            logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
        model.to(dev)

        # Calculate scores
        scored_datasets = score_slates(datasets, model, config)
        test_scores_per_model[config_path] = scored_datasets["test"]

        # Free up memory
        del model

    # True relevance scores
    true_test_scores = datasets["test"].y_by_qid

    # Max documents in a single query
    max_query_documents = max([y_true.size for y_true in true_test_scores])

    borda = Borda()
    copeland = Copeland()
    borda_rankings = []
    copeland_rankings = []

    for i, y_true in enumerate(true_test_scores):
        # Get predictions from each model and combine
        scores_to_combine = []
        #print(test_scores_per_model)
        for config_path, scores in test_scores_per_model.items():
            y_pred = scores[1][i]
            
            # Get rankings
            _, indices = y_pred.sort(descending=True, dim=-1)

            indices = indices.cpu().detach().numpy()[:len(y_true)]            
            scores_to_combine.append(indices)
        
        borda_ranking = borda.combine(scores_to_combine)
        borda_rankings.append(borda_ranking)

        copeland_ranking = copeland.combine(scores_to_combine)
        copeland_rankings.append(copeland_ranking)

    # Pad scores to max documents in a query
    borda_scores = np.zeros((len(true_test_scores),max_query_documents))
    for i, (_, scores) in enumerate(borda_rankings):
        scores = np.pad(np.array(scores), (0,max_query_documents-len(scores)), mode="constant", constant_values=(0,-1))
        borda_scores[i] = scores

    copeland_scores = np.zeros((len(true_test_scores),max_query_documents))
    for i, (_, scores) in enumerate(copeland_rankings):
        scores = np.pad(np.array(scores), (0,max_query_documents-len(scores)), mode="constant", constant_values=(0,-1))
        copeland_scores[i] = scores
    
    true_test_scores = np.array([np.pad(true_scores, (0,max_query_documents-true_scores.size), mode="constant", constant_values=(0,-1)) for true_scores in true_test_scores])
    
    # Calculate NDCG and MRR
    print(torch.mean(allrank_metrics.ndcg(torch.from_numpy(borda_scores), torch.from_numpy(true_test_scores),padding_indicator=-1, ats=ATS),dim=0))
    print(torch.mean(allrank_metrics.ndcg(torch.from_numpy(copeland_scores), torch.from_numpy(true_test_scores),padding_indicator=-1, ats=ATS),dim=0))
    print(torch.mean(allrank_metrics.mrr(torch.from_numpy(borda_scores), torch.from_numpy(true_test_scores),padding_indicator=-1, ats=ATS),dim=0))
    print(torch.mean(allrank_metrics.mrr(torch.from_numpy(copeland_scores), torch.from_numpy(true_test_scores),padding_indicator=-1, ats=ATS),dim=0))

if __name__ == "__main__":
    run()