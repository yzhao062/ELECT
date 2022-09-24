# -*- coding: utf-8 -*-
# coverage-driven model initialization

import os
import pandas as pd
import numpy as np
from scipy.stats import norm, rankdata
from scipy.stats import kendalltau
import itertools
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils import shuffle
from copy import deepcopy


def get_model_family(n_family, n_models):
    base_detectors_ranges = {}
    base_detectors_maps = {}
    
    
    keys = list(range(n_family))
    base_detectors_ranges[0] = list(range(0, 54))  # LODA
    base_detectors_ranges[1] = list(range(54, 61))  # ABOD
    base_detectors_ranges[2] = list(range(61, 142))  # iForest
    base_detectors_ranges[3] = list(range(142, 178))  # kNN
    base_detectors_ranges[4] = list(range(178, 214))  # LOF
    base_detectors_ranges[5] = list(range(214, 254))  # HBOS
    base_detectors_ranges[6] = list(range(254, 290))  # OCSVM
    base_detectors_ranges[7] = list(range(290, 297))  # COF
    
    keysm = list(range(n_models))
    for k in keys:
        keyys = base_detectors_ranges[k]
        for l in keyys:
            base_detectors_maps[l] = k
            
    return base_detectors_ranges, base_detectors_maps

# need to fix n_datasets 
ap_full = pd.read_excel(os.path.join("intermediate_files", 'AP_full.xlsx'), engine="openpyxl")

ap_values = ap_full.to_numpy()[:, 1:]

datasets = ap_full['Dataset'].tolist()
models = np.asarray(ap_full.columns.tolist()[1:])



n_datasets = ap_full.shape[0]
n_models = len(models)
n_family = 8
random_seed = 42


ap_ranks = rankdata(ap_values*-1, axis=1)
model_ap_var = np.var(ap_ranks, axis=0)
sorted_index = np.argsort(model_ap_var)
# this mapping is used for initial model selection
base_detectors_ranges, base_detectors_maps = get_model_family(n_family, n_models)
        

# initialize random pool
initial_models = []
all_models = list(range(n_models))


# increasing this number for more candidates
k = 20
cd = k

n_initial_model_max = 8
improvement_cut = 3

models_topk = {}
models_botk = {}

models_topk_len = []
models_botk_len = []

# do not consider these 
all_zeros = []
one_zeros = []
non_zeros = []

uncoverd_datasets_top = list(range(n_datasets))
uncoverd_datasets_bot = list(range(n_datasets))

coverd_datasets_top = []
coverd_datasets_bot = []

eps = 0

for i in range(ap_ranks.shape[1]):
    
    # note this does not equal to k due to tie
    models_topk[i] = np.where(ap_ranks[:, i] <= k)[0]
    models_topk_len.append(len(models_topk[i])+eps)
    
    models_botk[i] = np.where(ap_ranks[:, i] >= n_models-k)[0]
    models_botk_len.append(len(models_botk[i])+eps)

    if models_topk_len[-1]==eps and models_botk_len[-1] == eps:
        all_zeros.append(i)
    elif models_topk_len[-1]==eps or models_botk_len[-1] == eps:
        one_zeros.append(i)
    else:
        non_zeros.append(i)

print('selection candiate', models[non_zeros])
non_zeros_idx = np.asarray(deepcopy(non_zeros))

models_topk_len = np.asarray(models_topk_len)
models_botk_len = np.asarray(models_botk_len)
# np.asarray(models_topk_len)

# asymetric
valid_ratio1 = models_topk_len[non_zeros] / models_botk_len[non_zeros] 
valid_ratio2 = models_botk_len[non_zeros] / models_topk_len[non_zeros] 

valid_ratio = 0.5*(valid_ratio1+valid_ratio2)

# smaller the better
valid_diff = np.abs(valid_ratio - 1)


for j in range(100):
    improve_count = 0
    
    ## find the smallest one if not none
    ## we only need ths smallest one-> balance first and then covering
    candidates = np.argsort(valid_diff)[:cd]
    
    act_idx = non_zeros_idx[candidates]
    
    
    act_idx_gain = np.zeros([n_models])
    for idx in act_idx:
        for dt in uncoverd_datasets_top:
            if dt in models_topk[idx]:
                act_idx_gain[idx] +=1
                
        for dt in uncoverd_datasets_bot:
            if dt in models_botk[idx]:
                act_idx_gain[idx] +=1
    
    curr_max = np.argmax(act_idx_gain)
    
    # pop from the candiate by setting its valid ratio to a large number
    # handle anyway whether selected or not
    valid_diff[np.where(non_zeros_idx==curr_max)[0]] = 9999
    
    for c in models_topk[curr_max]:
        # count improvement
        if c in uncoverd_datasets_top:
            improve_count += 1

    
    for c in models_botk[curr_max]:
        # count improvement
        if c in uncoverd_datasets_bot:
            improve_count += 1
    
    if improve_count >= improvement_cut:         
        initial_models.append(curr_max)
        print('iter', j, 'improvement', improve_count, 'selecting', models[curr_max])
        
        # update cover and uncover sets
        for c in models_topk[curr_max]:
            # update covered and uncoverd set
            if c in uncoverd_datasets_top:
                uncoverd_datasets_top.remove(c)
                coverd_datasets_top.append(c)
                improve_count += 1
                print(j, 'removeing dataset', c, 'from uncoverd top')
    
        
        for c in models_botk[curr_max]:
            # update covered and uncoverd set
            if c in uncoverd_datasets_bot:
                uncoverd_datasets_bot.remove(c)
                coverd_datasets_bot.append(c)
                improve_count += 1
                print(j, 'removeing dataset', c, 'from uncoverd bot')
    else:
        continue
            
    print()
    
    if len(initial_models) == n_initial_model_max:
        break
    
    if len(uncoverd_datasets_top) == 0 and  len(uncoverd_datasets_bot)==0:
        break

# we may not need to cover all!
print(initial_models)    
print(models[initial_models])
print(len(uncoverd_datasets_bot), len(uncoverd_datasets_top))

print()
print('uncovered bottom datasets', np.asarray(datasets)[uncoverd_datasets_bot])
print()
print('uncovered top datasets', np.asarray(datasets)[uncoverd_datasets_top])