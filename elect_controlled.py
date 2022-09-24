# -*- coding: utf-8 -*-
# reproducible code for the controlled testbed

import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import kendalltau
import itertools
import lightgbm as lgb
from sklearn.utils import shuffle
from scipy.stats import rankdata

from utility import get_model_family, get_diff, get_cv_fold, get_initial_models
from utility import get_dataset_similarity, get_normalized_ap_diff, \
    weighted_kendall_from_pairs, get_rank

ap_full = pd.read_csv(os.path.join("intermediate_files", 'AP_controlled.csv'))

ap_score = ap_full.to_numpy()[:, 4:].astype(float)

datasets = ap_full['Data'].tolist()
models = ap_full.columns.tolist()[4:]

n_datasets = len(datasets)
n_models = len(models)
n_family = 8
random_seed = 42

# raw ap values 
ap_values = ap_full.to_numpy()[:, 4:]
ap_ranks = rankdata(ap_values, axis=1)

# this mapping is used for initial model selection
base_detectors_ranges, base_detectors_maps = get_model_family(n_family,
                                                              n_models)

# ##load X,y
X = np.load(os.path.join("intermediate_files", 'X_controlled.npy'))
y = np.load(
    os.path.join("intermediate_files", 'y_controlled.npy'))  # for regression
internel_measure_mat = np.load(
    os.path.join("intermediate_files", 'imp_controlled.npy'))

full_range = list(range(297))

# this will create 43956 pairs 
pairs = list(itertools.combinations(full_range, 2))

random_state = np.random.RandomState(random_seed)
all_data_index = list(range(n_datasets))
fold_index_list, data_headers_random = get_cv_fold(n_datasets, n_datasets,
                                                   datasets,
                                                   random_state=random_state)

# the selected results
dataset_header = []
selected_ap_header = []

n_sim_datasets = 5
n_iter = 50

# for tracking ei changes and best test ap change
ei_tracker = np.zeros([n_datasets, n_iter])
best_ap_tracker = np.zeros([n_datasets, n_iter])
avg_ap_sim_tracker = np.zeros([n_datasets, n_iter])
rank_tracker = np.zeros([n_datasets, n_iter])

model_history = np.zeros([n_datasets, n_iter]).astype('object')
model_history_raw = np.zeros([n_datasets, n_iter])
average_wk_sim_history = np.zeros([n_datasets, n_iter]).astype('float')
positive_count_history = np.zeros([n_datasets, n_iter]).astype('float')
keep_pairs_history = np.zeros([n_datasets, n_iter]).astype('float')

avg_dataset_similarity_header = []
reg_dataset_similarity_header = []
reg_dataset_similarity_all = []
selected_datasets_similarity = []

# cheat_datasets = [5, 17, 18, 30, 31, 33, 34, 35, 38]
for ind, data_ind in enumerate(fold_index_list):

    neighbor_counter = np.zeros([n_datasets, ])

    test_index = fold_index_list[ind]
    train_index = get_diff(all_data_index,
                           test_index)  # subtract test from all

    ###########################################################################
    # build lightgbm using the training data

    test_lightgbm_ind = list(
        range(test_index[0] * len(pairs), (test_index[0] + 1) * len(pairs)))
    train_lightgbm_ind = get_diff(list(range(n_datasets * len(pairs))),
                                  test_lightgbm_ind)
    assert ((len(test_lightgbm_ind) + len(
        train_lightgbm_ind)) == n_datasets * len(pairs))

    X_train_lightgbm = X[train_lightgbm_ind, :]
    y_train_lightgbm = y[train_lightgbm_ind]

    X_train_lightgbm, y_train_lightgbm = shuffle(X_train_lightgbm,
                                                 y_train_lightgbm,
                                                 random_state=42)
    clf = lgb.LGBMRegressor(n_jobs=4, random_state=42,
                            num_leaves=16,
                            max_depth=-1,
                            learning_rate=0.01,
                            n_estimators=200,
                            objective='huber',
                            min_data_in_leaf=1000)

    clf.fit(X_train_lightgbm, y_train_lightgbm)

    ###########################################################################
    all_models = list(range(n_models))
    # decided by the coverage initialization
    initial_models = [142, 179, 286, 143, 243, 180, 288, 0]

    all_models = list(range(n_models))

    print(initial_models)
    curr_models = initial_models
    left_models = get_diff(all_models, curr_models)

    stop_counter = 0
    for j in range(n_iter):
        # turn itself as -1 to prevent selection
        dataset_similarity = np.zeros([n_datasets, ])
        dataset_similarity[test_index] = -1

        dataset_similarity_i = np.zeros([n_datasets, ])
        dataset_similarity_i[test_index] = -1
        ##################################################################################################
        # find the data similarity of all train datasets
        for k in train_index:
            dataset_similarity[k] = get_dataset_similarity(
                ap_values[k, curr_models],
                ap_values[test_index, curr_models])
        ##############################################################################################

        curr_pairs = list(
            itertools.combinations(list(range(len(curr_models))), 2))
        curr_pairs_numpy = np.asarray(curr_pairs)

        # ***********************************************************************************************
        test_internal_measure = internel_measure_mat[
                                test_index[0] * 3:(test_index[0] + 1) * 3,
                                curr_models]
        test_lightgbm_mat = np.zeros([len(curr_pairs), 9])
        test_lightgbm_mat_reverse = np.zeros([len(curr_pairs), 9])
        for cpi, cp in enumerate(curr_pairs):
            test_lightgbm_mat[cpi, :] = np.concatenate(
                (test_internal_measure[:, cp[0]],
                 test_internal_measure[:, cp[1]],
                 test_internal_measure[:, cp[0]] - test_internal_measure[:,
                                                   cp[1]])).reshape(1, -1)

            test_lightgbm_mat_reverse[cpi, :] = np.concatenate(
                (test_internal_measure[:, cp[1]],
                 test_internal_measure[:, cp[0]],
                 test_internal_measure[:, cp[1]] - test_internal_measure[:,
                                                   cp[0]])).reshape(1, -1)
        # hard threshold is no longer needed 
        test_pair_pred = clf.predict(test_lightgbm_mat)
        # test_ap_norm = get_normalized_ap_diff(test_pair_pred)

        ###################################################################
        test_pair_pred_reverse = clf.predict(test_lightgbm_mat_reverse)
        test_pair_pred_comb = 0.5 * (
                    test_pair_pred + np.negative(test_pair_pred_reverse))

        contradicted_pairs = np.where(
            np.sign(test_pair_pred) != np.sign(test_pair_pred_reverse * -1))[
            0].tolist()
        kept_pairs = get_diff(list(range(len(curr_pairs))), contradicted_pairs)
        test_ap_norm = get_normalized_ap_diff(test_pair_pred_comb[kept_pairs])

        keep_sim, cont_sim, all_sim = [], [], []
        for k in train_index:
            train_value_curr = ap_values[k, curr_models]
            train_a = train_value_curr[curr_pairs_numpy[:, 0]]
            train_b = train_value_curr[curr_pairs_numpy[:, 1]]
            train_ap_diff = train_a - train_b

            train_ap_norm = get_normalized_ap_diff(train_ap_diff[kept_pairs])

            dataset_similarity_i[k] = weighted_kendall_from_pairs(test_ap_norm,
                                                                  train_ap_norm)

        print('kendall between ours and ground truth',
              get_dataset_similarity(dataset_similarity, dataset_similarity_i))


        # identify the most similar datasets
        index_sorted = np.argsort(dataset_similarity_i * -1)
        similar_datasets = index_sorted[:n_sim_datasets]

        neighbor_counter[similar_datasets] = neighbor_counter[
                                                 similar_datasets] + 1
        # print(neighbor_counter)

        # similar_datasets = np.where(neighbor_counter>j*0.5)[0].tolist()

        actual_similar_datasets = np.argsort(dataset_similarity * -1)[
                                  :n_sim_datasets]

        # get the current best model average ap on the similar datasets
        curr_model_mean = []
        for m in curr_models:
            curr_model_mean.append(ap_values[similar_datasets, m].mean())
        curr_model_best = np.max(curr_model_mean)
        best_model_so_far = curr_models[np.argmax(curr_model_mean)]

        mu_list = []
        sigma_list = []
        for m in left_models:
            ap_model = ap_values[similar_datasets, m]
            mu, sigma = np.mean(ap_model), np.std(ap_model)
            mu_list.append(mu)
            sigma_list.append(sigma)

        z_list = (mu_list - curr_model_best) / sigma_list
        ei = (mu_list - curr_model_best) * norm.cdf(
            z_list) + sigma_list * norm.pdf(z_list)
        ei[np.where(sigma_list == 0)] = 0

        # next best model
        ei_max = np.argmax(ei)
        next_model = left_models[ei_max]

        # add to current model, remove from left model
        curr_models.append(next_model)
        left_models.remove(next_model)

        assert (len(curr_models) + len(left_models) == n_models)
        model_history_raw[test_index, j] = next_model

        # get the current best model average ap on the similar datasets
        # only use the evaluated model
        curr_model_mean = []

        # select from all models
        for m in all_models:
            curr_model_mean.append(ap_values[similar_datasets, m].mean())

            # # this does not work
            # curr_model_mean.append(ap_ranks[similar_datasets, m].mean())
        best_model_so_far = all_models[np.argmax(curr_model_mean)]

        print(test_index[0], datasets[test_index[0]],
              'after selected test AP',
              ap_values[test_index, best_model_so_far][0],
              '| average ap',
              np.round(ap_values[test_index,].mean(), decimals=4),
              '| best ap',
              np.round(ap_values[test_index,].max(), decimals=4), )
        print('iter', j, 'selected model', models[best_model_so_far],
              'model rank', 298 - ap_ranks[test_index, best_model_so_far])
        print('actual neighbors:',
              np.sort(np.asarray(datasets)[actual_similar_datasets]))
        print('selected neighbors:',
              np.sort(np.asarray(datasets)[similar_datasets]))
        print()
        model_history[test_index, j] = models[best_model_so_far]

        keep_pairs_history[test_index, j] = len(kept_pairs) / len(curr_pairs)

        # track the choice based on the current choice
        ei_tracker[test_index[0], j] = np.max(ei)
        best_ap_tracker[test_index[0], j] = \
        ap_values[test_index, best_model_so_far][0]
        rank_tracker[test_index[0], j] = 298 - ap_ranks[
            test_index, best_model_so_far]

        avg_ap_sim_tracker[test_index[0], j] = dataset_similarity[
            similar_datasets].mean()

    similar_pairs = list(itertools.combinations(similar_datasets, 2))
    similar_pairs_list = []

    selected_datasets_similarity.append(similar_pairs_list)

    # actual wk to test by ground truth
    # avg_dataset_similarity_header.append(all_dataset_similarity[test_index, similar_datasets])
    # regressor believe wk to test 
    reg_dataset_similarity_header.append(
        dataset_similarity_i[similar_datasets])
    # regressor believe wk to teston all datasets
    reg_dataset_similarity_all.append(dataset_similarity_i)

    # store ground truth
    dataset_header.append(datasets[data_ind[0]])
    selected_ap_header.append(ap_values[test_index, best_model_so_far][0])

# save our result
result = np.column_stack((dataset_header, selected_ap_header))
result_df = pd.DataFrame(result, columns=['Dataset', 'ap'])

# %%
# statistical test
from scipy.stats import wilcoxon

# make sure the datasets are comoarable
sort_by_name = ap_score
random_baseline = np.mean(sort_by_name, axis=1)
sorted_ap_val = np.sort(sort_by_name, axis=1)
sorted_ap_val_1d = np.mean(sorted_ap_val, axis=0)
iforest_baseline = np.mean(sort_by_name[:, base_detectors_ranges[2]],
                           axis=1).astype(float)

our_result = result_df['ap'].astype(float).tolist()

our_qth = 0
iforest_qth = int(np.mean(get_rank(ap_score, iforest_baseline)))
random_qth = int(np.mean(get_rank(ap_score, random_baseline)))

for i in range(n_models):
    print('top', i + 1,
          wilcoxon(sorted_ap_val[:, 296 - i], our_result,
                   alternative='greater'), np.mean(sorted_ap_val[:, 296 - i]),
          np.mean(our_result))
    if wilcoxon(sorted_ap_val[:, 296 - i], our_result, alternative='greater')[
        1] >= 0.05:
        our_qth = i + 1
        break
