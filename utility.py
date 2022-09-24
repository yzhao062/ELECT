# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import rankdata, kendalltau


def get_rank(ap_score, our_result):
    all_ranks = rankdata(ap_score*-1, axis=1)
    
    rank_list = []
    
    for j in range(len(our_result)):
        rank_list.append(all_ranks[j, np.argmin(np.abs(ap_score[j, :]-our_result[j]))])
    
    return rank_list

def argmaxatn(w, nth):
    w = np.asarray(w).ravel()
    t = np.argsort(w)
    return t[-1*nth]


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

def get_diff(li1, li2): 
    return (list(set(li1) - set(li2))) 

def get_cv_fold(n_datasets, n_folds, datasets, random_state):

    all_data_index = list(range(n_datasets))
    # random_state = np.random.RandomState(random_seed)
    # random_state.shuffle(all_data_index)
    
    # n_folds = n_datasets  # LOO CV
    
    # n_datasets_fold = int(n_sub_datasets/n_folds)
    n_datasets_fold = int(n_datasets/n_folds)
    fold_index_list = []
    
    for i in range(n_folds):
        if i == n_folds-1:
            fold_index_list.append(all_data_index[i*n_datasets_fold:])
        else:
            fold_index_list.append(all_data_index[i*n_datasets_fold: (i+1)*n_datasets_fold])
    
    data_headers_random = []    
    for i in fold_index_list:
        data_headers_random.append(datasets[i[0]])
        
    return fold_index_list, data_headers_random

def get_initial_models(ap_values):
    # ap_full = pd.read_excel('AP_full.xlsx', engine="openpyxl")
    
    # datasets = ap_full['Dataset'].tolist()
    # models = ap_full.columns.tolist()[1:]
    
    # n_datasets = len(datasets)
    n_models = ap_values.shape[1]
    n_family = 8
    
    # raw ap values 
    # ap_values = ap_full.to_numpy()[:, 1:]
    ap_ranks = rankdata(ap_values*-1, axis=1)
    model_ap_var = np.var(ap_ranks, axis=0)
    # this mapping is used for initial model selection
    base_detectors_ranges, base_detectors_maps = get_model_family(n_family, n_models)
            
    
    # initialize random pool
    initial_models = []
    all_models = list(range(n_models))
    
    # # one strategy is to randomly select from each family
    for i in range(n_family):
        # initial_models.extend(random_state.choice(base_detectors_ranges[i], 2, replace=False))
        # initial_models.append(random_state.choice(base_detectors_ranges[i]))
        family_model_max_idx = np.argmax(model_ap_var[base_detectors_ranges[i]])
        initial_models.append(base_detectors_ranges[i][family_model_max_idx])
    
    return initial_models

def get_initial_models16(ap_values):
    # ap_full = pd.read_excel('AP_full.xlsx', engine="openpyxl")
    
    # datasets = ap_full['Dataset'].tolist()
    # models = ap_full.columns.tolist()[1:]
    
    # n_datasets = len(datasets)
    n_models = ap_values.shape[1]
    n_family = 8
    
    # raw ap values 
    # ap_values = ap_full.to_numpy()[:, 1:]
    ap_ranks = rankdata(ap_values*-1, axis=1)
    model_ap_var = np.var(ap_ranks, axis=0)
    # this mapping is used for initial model selection
    base_detectors_ranges, base_detectors_maps = get_model_family(n_family, n_models)
            
    
    # initialize random pool
    initial_models = []
    all_models = list(range(n_models))
    
    # # one strategy is to randomly select from each family
    for i in range(n_family):
        # initial_models.extend(random_state.choice(base_detectors_ranges[i], 2, replace=False))
        # initial_models.append(random_state.choice(base_detectors_ranges[i]))
        family_model_max_idx = np.argmax(model_ap_var[base_detectors_ranges[i]])
        sorted_index = np.argsort(model_ap_var[base_detectors_ranges[i]]*-1).tolist()
        # print(sorted_index)
        # print(base_detectors_ranges[i][sorted_index[0:2]])
        initial_models.append(base_detectors_ranges[i][sorted_index[0]])
        initial_models.append(base_detectors_ranges[i][sorted_index[1]])
    
    return initial_models

def get_permutations(list1, list2):
    full_perm = []
    for i in list1:
        for j in list2:
            full_perm.append((i, j))
            
    return full_perm

def get_dataset_similarity(ap1, ap2):
    # should write as pairwise, but use scipy for now
    return kendalltau(ap1, ap2)[0]

def get_dataset_similarity_pair(pair_list):
    return np.sum(pair_list)/len(pair_list)


def get_normalized_ap_diff(ap_diff):
    max_ap_diff = np.max(np.abs(ap_diff))
    return ap_diff/(max_ap_diff+0.00000001)

def weighted_kendall_from_pairs(a, b):
    c1_ind = np.abs(a)<=np.abs(b)
    c2_ind = np.abs(a)>np.abs(b)
    c1 = a/(b+0.0000001)
    c2 = b/(a+0.0000001) 
    c = np.zeros([len(a),])
    c[c1_ind] = c1[c1_ind]
    c[c2_ind] = c2[c2_ind]
    
    return np.sum(c)/np.sum(np.abs(c))

# def get_initial_models():
#     ap_full = pd.read_excel('AP_full.xlsx', engine="openpyxl")
    
#     datasets = ap_full['Dataset'].tolist()
#     models = ap_full.columns.tolist()[1:]
    
#     n_datasets = len(datasets)
#     n_models = len(models)
#     n_family = 8
#     random_seed = 42
    
#     # raw ap values 
#     ap_values = ap_full.to_numpy()[:, 1:]
#     ap_ranks = rankdata(ap_values*-1, axis=1)
#     model_ap_var = np.var(ap_ranks, axis=0)
#     # this mapping is used for initial model selection
#     base_detectors_ranges, base_detectors_maps = get_model_family(n_family, n_models)
            
    
#     # initialize random pool
#     initial_models = []
#     all_models = list(range(n_models))
    
#     # # one strategy is to randomly select from each family
#     for i in range(n_family):
#         # initial_models.extend(random_state.choice(base_detectors_ranges[i], 2, replace=False))
#         # initial_models.append(random_state.choice(base_detectors_ranges[i]))
#         family_model_max_idx = np.argmax(model_ap_var[base_detectors_ranges[i]])
#         initial_models.append(base_detectors_ranges[i][family_model_max_idx])
    
#     return initial_models


#%%

# import os
# import pandas as pd
# import numpy as np
# from scipy.stats import norm, rankdata
# from scipy.stats import kendalltau
# import itertools
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.utils import shuffle



# ap_full = pd.read_excel('AP_full.xlsx', engine="openpyxl")

# datasets = ap_full['Dataset'].tolist()
# models = np.asarray(ap_full.columns.tolist()[1:])

# n_datasets = len(datasets)
# n_models = len(models)
# n_family = 8
# random_seed = 42

# # raw ap values 
# ap_values = ap_full.to_numpy()[:, 1:]

# n_models = ap_values.shape[1]
# n_family = 8

# # raw ap values 
# # ap_values = ap_full.to_numpy()[:, 1:]
# ap_ranks = rankdata(ap_values*-1, axis=1)
# model_ap_var = np.var(ap_ranks, axis=0)
# sorted_index = np.argsort(model_ap_var)
# # this mapping is used for initial model selection
# base_detectors_ranges, base_detectors_maps = get_model_family(n_family, n_models)
        

# # initialize random pool
# initial_models = []
# all_models = list(range(n_models))

# # # # one strategy is to randomly select from each family
# # for i in range(n_family):
# #     # initial_models.extend(random_state.choice(base_detectors_ranges[i], 2, replace=False))
# #     # initial_models.append(random_state.choice(base_detectors_ranges[i]))
# #     family_model_max_idx = np.argmax(model_ap_var[base_detectors_ranges[i]])
# #     # initial_models.append(base_detectors_ranges[i][family_model_max_idx])
# #     sorted_index = np.argsort(model_ap_var[base_detectors_ranges[i]]*-1).tolist()
    
# #     initial_models.append(base_detectors_ranges[i][sorted_index[0]])
# #     initial_models.append(base_detectors_ranges[i][sorted_index[1]])
# #     # initial_models.append(base_detectors_ranges[i][sorted_index[2]])

# # another strategy is to select from all with the largest variance

# initial_models = sorted_index[-16:].tolist()

# models_comb = list(itertools.combinations(initial_models, 8))
# pairs = list(itertools.combinations(list(range(8)), 2))

# model_mean_wk = []
# for idx, modelc in enumerate(models_comb):
#     total_sim = []
#     for p in pairs:
#         total_sim.append(kendalltau(ap_ranks[:, modelc[p[0]]], ap_ranks[:, modelc[p[1]]])[0])
#     model_mean_wk.append(np.mean(total_sim))
#     print(idx, model_mean_wk[-1])
    
# np.min(model_mean_wk)
# np.argmin(model_mean_wk)
    
# return initial_models

# #%%

# import os
# import pandas as pd
# import numpy as np
# from scipy.stats import norm, rankdata
# from scipy.stats import kendalltau
# import itertools
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.utils import shuffle
# from copy import deepcopy


# def get_model_family(n_family, n_models):
#     base_detectors_ranges = {}
#     base_detectors_maps = {}
    
    
#     keys = list(range(n_family))
#     base_detectors_ranges[0] = list(range(0, 54))  # LODA
#     base_detectors_ranges[1] = list(range(54, 61))  # ABOD
#     base_detectors_ranges[2] = list(range(61, 142))  # iForest
#     base_detectors_ranges[3] = list(range(142, 178))  # kNN
#     base_detectors_ranges[4] = list(range(178, 214))  # LOF
#     base_detectors_ranges[5] = list(range(214, 254))  # HBOS
#     base_detectors_ranges[6] = list(range(254, 290))  # OCSVM
#     base_detectors_ranges[7] = list(range(290, 297))  # COF
    
#     keysm = list(range(n_models))
#     for k in keys:
#         keyys = base_detectors_ranges[k]
#         for l in keyys:
#             base_detectors_maps[l] = k
            
#     return base_detectors_ranges, base_detectors_maps

# # need to fix n_datasets 

# ap_full = pd.read_excel('AP_full.xlsx', engine="openpyxl")

# # datasets = ap_full['Dataset'].tolist()
# models = np.asarray(ap_full.columns.tolist()[1:])

# # n_datasets = len(datasets)

# n_datasets = ap_full.shape[0]
# n_models = len(models)
# n_family = 8
# random_seed = 42

# # raw ap values 
# ap_values = ap_full.to_numpy()[:, 1:]

# n_models = ap_values.shape[1]
# n_family = 8

# # raw ap values 
# # ap_values = ap_full.to_numpy()[:, 1:]
# ap_ranks = rankdata(ap_values*-1, axis=1)
# model_ap_var = np.var(ap_ranks, axis=0)
# sorted_index = np.argsort(model_ap_var)
# # this mapping is used for initial model selection
# base_detectors_ranges, base_detectors_maps = get_model_family(n_family, n_models)
        

# # initialize random pool
# initial_models = []
# all_models = list(range(n_models))


# # increasing this number for more candidates
# k = 20

# models_topk = {}
# models_botk = {}

# models_topk_len = []
# models_botk_len = []

# # do not consider these 
# all_zeros = []
# one_zeros = []
# non_zeros = []

# uncoverd_datasets_top = list(range(n_datasets))
# uncoverd_datasets_bot = list(range(n_datasets))

# coverd_datasets_top = []
# coverd_datasets_bot = []

# eps = 0

# for i in range(ap_ranks.shape[1]):
#     # print(np.where(ap_ranks[:, i] <= k)[0])
#     # track each models outperforming datasets
    
#     # note this does not equal to k due to tie
#     models_topk[i] = np.where(ap_ranks[:, i] <= k)[0]
#     models_topk_len.append(len(models_topk[i])+eps)
    
#     models_botk[i] = np.where(ap_ranks[:, i] >= n_models-k)[0]
#     models_botk_len.append(len(models_botk[i])+eps)

#     if models_topk_len[-1]==eps and models_botk_len[-1] == eps:
#         all_zeros.append(i)
#     elif models_topk_len[-1]==eps or models_botk_len[-1] == eps:
#         one_zeros.append(i)
#     else:
#         non_zeros.append(i)

# print('selection candiate', models[non_zeros])
# non_zeros_idx = np.asarray(deepcopy(non_zeros))

# models_topk_len = np.asarray(models_topk_len)
# models_botk_len = np.asarray(models_botk_len)
# # np.asarray(models_topk_len)

# # asymetric
# valid_ratio1 = models_topk_len[non_zeros] / models_botk_len[non_zeros] 
# valid_ratio2 = models_botk_len[non_zeros] / models_topk_len[non_zeros] 

# valid_ratio = 0.5*(valid_ratio1+valid_ratio2)

# # smaller the better
# valid_diff = np.abs(valid_ratio - 1)


# candidate_size = 30
# for j in range(15):
#     improve_count = 0
    
#     # find the smallest one if not none
#     unique, counts = np.unique(valid_diff, return_counts=True)

#     # we only need ths smallest one-> balance first and then covering
#     # candidates = np.where(valid_diff == unique[0])[0]
    
#     # 也可以sort之后选最小的
#     candidates = np.argsort(valid_diff)[:candidate_size]
    
    
#     act_idx = non_zeros_idx[candidates]
    
#     act_idx_gain = np.zeros([n_models])
#     for idx in act_idx:
#         for dt in uncoverd_datasets_top:
#             if dt in models_topk[idx]:
#                 act_idx_gain[idx] +=1
                
#         for dt in uncoverd_datasets_bot:
#             if dt in models_botk[idx]:
#                 act_idx_gain[idx] +=1
    
#     curr_max = np.argmax(act_idx_gain)
    
#     # pop from the candiate by setting its valid ratio to a large number
#     valid_diff[np.where(non_zeros_idx==curr_max)[0]] = 9999
    
#     for c in models_topk[curr_max]:
#         # update covered and uncoverd set
#         if c in uncoverd_datasets_top:
#             uncoverd_datasets_top.remove(c)
#             coverd_datasets_top.append(c)
#             improve_count += 1
#             print(j, 'removeing dataset', c, 'from uncoverd top')

    
#     for c in models_botk[curr_max]:
#         # update covered and uncoverd set
#         if c in uncoverd_datasets_bot:
#             uncoverd_datasets_bot.remove(c)
#             coverd_datasets_bot.append(c)
#             improve_count += 1
#             print(j, 'removeing dataset', c, 'from uncoverd bot')
    
#     if improve_count > 0:         
#         initial_models.append(curr_max)
#         # 可以增大  candidate_size cover的数量越来越少
        
#     print()
    
    
#     if len(uncoverd_datasets_top) == 0 and  len(uncoverd_datasets_bot)==0:
#         break

# # we may not need to cover all!
# print(initial_models)    
