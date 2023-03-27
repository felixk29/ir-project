# Allrank configs for IR

This folder contains the configs files use to run experiments on the allrank framework, as well as the result of those experiments. The structure is as follows.

- `config/base_trains`: contains 10 config files, one per loss function and dataset combination. The model used is the one used in the NeuralNDCG paper, available in the reproductibility folder of allrank.

- `data`: contains a folder for each of the datasets. The MQ2008 is provided as example of the structure, the MSWEB10K is not uploaded due to size. To use this dataset, simply copy the contents of a Fold into a folder called `web10k`.

- `runs`: results of the runs done in allrank.

- `train_all.sh`: script to run every config file in a folder.

## How to install and run allrank

Here are the instructions to install allrank and run the config files. It is recommended to use a Linux distro, but WSL should work too. It is also recommended to have a GPU and CUDA installed, or some complex models will take a lot of time to train.

1. Install Python 3.8. There has been compatibility problems with other versions.
2. Clone the allrank repo locally.
3. In the `setup.py` file, change line 20 to `google-auth==2.15.0`. It gives an error during installation otherway.
4. Install necessary dependencies using `make install-reqs`.
5. Copy the `train_all.sh` script to the allrank repo folder, and also create a folder with config files and the data.
6. Train all config files in a folder using `./train_all.sh config-path results-path`.

For example, with the provided config, just move the script, data* and configs folder to the allrank folder, and run `./train_all.sh ./configs/base_trains/ ./results/base_trains/`.

*Note: You will need to change the data path in the JSON config files, since it is given as an absolute path to the folder.
