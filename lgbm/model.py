import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_svmlight_file
import allrank.models.losses as aLosses
import torch as t
from torch import Tensor

def grouping(x):
  xn=np.unique(x,return_counts=True,return_index=True)
  return xn[2][xn[1].argsort()]

# load mq dataset into lgb.Dataset
print('loading mq2008 dataset')
mq_tr = load_svmlight_file('data/mq2008/train.txt', query_id=True,)
mq_train= lgb.Dataset(mq_tr[0], mq_tr[1], group=grouping(mq_tr[2]),free_raw_data=False)

mq_ts = load_svmlight_file('data/mq2008/test.txt',query_id=True)
mq_test = lgb.Dataset(mq_ts[0], mq_ts[1], group=grouping(mq_ts[2]),free_raw_data=False)

mq_val = load_svmlight_file('data/mq2008/vali.txt',query_id=True)
mq_valid = lgb.Dataset(mq_val[0], mq_val[1], group=grouping(mq_val[2]),free_raw_data=False)

# #load web10k dataset into lgb.Dataset 
print('loading web10k dataset') 
web_tr = load_svmlight_file('data/mslr-web10k/Fold1/train.txt',query_id=True) 
web_train= lgb.Dataset(web_tr[0], web_tr[1], group=grouping(web_tr[2]))

web_ts = load_svmlight_file('data/mslr-web10k/Fold1/test.txt',query_id=True)
web_test = lgb.Dataset(web_ts[0], web_ts[1], group=grouping(web_ts[2]))

web_val = load_svmlight_file('data/mslr-web10k/Fold1/vali.txt',query_id=True)
web_valid = lgb.Dataset(web_val[0], web_val[1], group=grouping(web_val[2]))

mq_params = {'metric': ['ndcg','map'],
              'ndcg_eval_at': [5,10,30,60], # 10, 30, 60],
              'force_row_wise': True,
              'num_leaves': 40,
              'max_depth': 5,
              'min_data_in_leaf': 150,
              'num_iterations': 400,
              'learning_rate': 0.5,
              'early_stopping_round':100,
              'num_iterations': 400,
              }

web_params = {'metric': ['ndcg','map'],
              'ndcg_eval_at': [5,10,30,60], # 10, 30, 60],
              'force_row_wise': True,
              'num_leaves': 40,
              'max_depth': 5,
              'min_data_in_leaf': 150,
              'num_iterations': 400,
              'learning_rate': 0.5,
              'early_stopping_round':100,
              'num_iterations': 50007,
              }

models={}
print('starting model training')
#for dt in ['web']:
for dt in ['mq','web']:
  obj='lambdarank'#'regression','lambdarank','rank_xendcg']:#,"all_approxNDCG","all_neuralNDCG", "all_rankNet"]:
  # gbm=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',force_row_wise=True)
  # gbm.fit(locals()[dt+'_tr'][0],locals()[dt+'_tr'][1],group=[1]*locals()[dt+'_tr'][0].shape[0])
  if dt=='mq':
    params=mq_params
  else:
    params=web_params

  params['objective']=obj
  gbm = lgb.train(params, locals()[dt+'_train'], verbose_eval=100,
                valid_sets=[locals()[dt+'_test'],locals()[dt+'_valid']], valid_names=['test','valid'])
  models[dt+'_'+obj]=gbm

import metrics

def mrr(model_name):
  model=models[model_name]
  if model_name.startswith('mq'):
    val=mq_val
    valid=mq_valid
  else:
    val=web_val
    valid=web_valid

  valid_data=[]
  valid_truths=[]
  current_i=0

  for i in valid.get_group():
    valid_data.append(val[0][current_i:current_i+i])
    valid_truths.append(val[1][current_i:current_i+i])
    current_i+=i
  mrrs=[]
  ndcgs=[]
  for i in range(len(valid_data)):
    pred=model.predict(valid_data[i])
    if len(pred)<60:
      pred=np.append(pred,np.ones(60-len(pred)*-1))
      truths=np.append(valid_truths[i],np.ones(60-len(valid_truths[i])*-1))
    else:
      truths=valid_truths[i]
    mrrs.append(metrics.mrr(t.Tensor([pred]),t.Tensor([truths]),[5,10,30,60],padding_indicator=-1))
    ndcgs.append(metrics.ndcg(t.Tensor([pred]),t.Tensor([truths]),[5,10,30,60],padding_indicator=-1))
  print(t.mean(t.stack(mrrs),dim=0))
  print(t.mean(t.stack(ndcgs),dim=0))


for i in models.keys():
  print(i)
  mrr(i)


