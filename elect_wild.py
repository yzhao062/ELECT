# -*- coding: utf-8 -*-
# reproducible code for the Wild testbed

import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import kendalltau
import itertools
import lightgbm as lgb
from sklearn.utils import shuffle
from scipy.stats import rankdata

from utility import get_model_family, get_diff, get_cv_fold, \
    get_initial_models, get_initial_models16

ap_full = pd.read_excel(os.path.join("intermediate_files", 'AP_full.xlsx'),
                        engine="openpyxl")

datasets = ap_full['Dataset'].tolist()
models = ap_full.columns.tolist()[1:]

n_datasets = len(datasets)
n_models = len(models)
n_family = 8
random_seed = 42

# raw ap values 
ap_values = ap_full.to_numpy()[:, 1:]
ap_ranks = rankdata(ap_values, axis=1)

# this mapping is used for initial model selection
base_detectors_ranges, base_detectors_maps = get_model_family(n_family,
                                                              n_models)

##load X,y
all_dataset_similarity = np.load(
    os.path.join("intermediate_files", "all_dataset_wk_similarity.npy"))
dataset_sim_ground_truth = (np.sum(all_dataset_similarity, axis=0) - 1) / 38

X = np.load(os.path.join("intermediate_files", 'X_wild.npy'))
y = np.load(os.path.join("intermediate_files", 'y_wild.npy'))  # for regression
internel_measure_mat = np.load(
    os.path.join("intermediate_files", 'imp_wild.npy'))


def get_dataset_similarity(ap1, ap2):
    # should write as pairwise, but use scipy for now
    return kendalltau(ap1, ap2)[0]


def get_dataset_similarity_pair(pair_list):
    return np.sum(pair_list) / len(pair_list)


def get_normalized_ap_diff(ap_diff):
    max_ap_diff = np.max(np.abs(ap_diff))
    return ap_diff / (max_ap_diff + 0.00000001)


def weighted_kendall_from_pairs(a, b):
    c1_ind = np.abs(a) <= np.abs(b)
    c2_ind = np.abs(a) > np.abs(b)
    c1 = a / (b + 0.0000001)
    c2 = b / (a + 0.0000001)
    c = np.zeros([len(a), ])
    c[c1_ind] = c1[c1_ind]
    c[c2_ind] = c2[c2_ind]

    return np.sum(c) / np.sum(np.abs(c))
    # return c


full_range = list(range(297))

# this will create 43956 pairs 
pairs = list(itertools.combinations(full_range, 2))

random_state = np.random.RandomState(random_seed)
all_data_index = list(range(n_datasets))
fold_index_list, data_headers_random = get_cv_fold(n_datasets, n_datasets,
                                                   datasets,
                                                   random_state=random_state)

# for tracking ei changes and best test ap change
ei_tracker = np.zeros([n_datasets, 289])
best_ap_tracker = np.zeros([n_datasets, 289])

# the selected results
dataset_header = []
selected_ap_header = []

n_sim_datasets = 5
n_iter = 50
conv_iter = 50

model_history = np.zeros([n_datasets, n_iter]).astype('object')
average_wk_sim_history = np.zeros([n_datasets, n_iter]).astype('float')
positive_count_history = np.zeros([n_datasets, n_iter]).astype('float')
keep_pairs_history = np.zeros([n_datasets, n_iter]).astype('float')

curr_models_test_ap_history = []

avg_dataset_similarity_header = []
reg_dataset_similarity_header = []
reg_dataset_similarity_all = []
selected_datasets_similarity = []
rank_tracker = []

conv_tracker = np.full([39, 1], n_iter)
for ind, data_ind in enumerate(fold_index_list):

    curr_models_test_ap_tracker = []
    sim_dataset_conv_list = []

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
                            num_leaves=64,
                            max_depth=20,
                            learning_rate=0.01,
                            n_estimators=200,
                            objective='huber',
                            min_data_in_leaf=1000)

    clf.fit(X_train_lightgbm, y_train_lightgbm)

    ###########################################################################
    all_models = list(range(n_models))
    # get from coverage initialization
    initial_models = [179, 253, 22, 16, 54, 214, 273, 291]

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
        #######################################################################
        # find the data similarity of all train datasets
        for k in train_index:
            dataset_similarity[k] = get_dataset_similarity(
                ap_values[k, curr_models],
                ap_values[test_index, curr_models])
        #######################################################################

        curr_pairs = list(
            itertools.combinations(list(range(len(curr_models))), 2))
        curr_pairs_numpy = np.asarray(curr_pairs)

        # *********************************************************************
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
        similar_datasets = np.sort(similar_datasets)

        actual_similar_datasets = np.argsort(dataset_similarity * -1)[
                                  :n_sim_datasets]

        # get the current best model average ap on the similar datasets
        curr_model_mean = []
        for m in curr_models:
            curr_model_mean.append(ap_values[similar_datasets, m].mean())
        curr_model_best = np.max(curr_model_mean)
        best_model_so_far = curr_models[np.argmax(curr_model_mean)]

        # https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
        # http://krasserm.github.io/2018/03/21/bayesian-optimization/

        # get the ap on the similar datasets, and estimate mean and std of each 
        # unevaluaed model
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

        # get the current best model average ap on the similar datasets
        # only use the evaluated model
        curr_model_mean = []

        # select from all models
        for m in all_models:
            curr_model_mean.append(ap_values[similar_datasets, m].mean())

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
        average_wk_sim_history[test_index, j] = np.mean(
            all_dataset_similarity[test_index, similar_datasets])
        positive_count_history[test_index, j] = (all_dataset_similarity[
                                                     test_index, similar_datasets] > 0).astype(
            int).sum()
        keep_pairs_history[test_index, j] = len(kept_pairs) / len(curr_pairs)

        # track the choice based on the current choice
        ei_tracker[test_index[0], j] = np.max(ei)
        best_ap_tracker[test_index[0], j] = \
            ap_values[test_index, best_model_so_far][0]

        sim_dataset_conv_list.append(similar_datasets.tolist())

        # get the last conv iter similar datasets
        last_k_iter_sim = np.asarray(sim_dataset_conv_list[-1 * conv_iter:])
        compare_k_iter_sim = np.tile(similar_datasets, (conv_iter, 1))
        # print(last_k_iter_sim)
        # print(compare_k_iter_sim)

        curr_models_test_ap_tracker.append(ap_values[test_index, curr_models])
        # first few iteration
        if last_k_iter_sim.shape[0] != compare_k_iter_sim.shape[0]:
            continue

        if np.sum(last_k_iter_sim == compare_k_iter_sim) == int(
                conv_iter * n_sim_datasets):
            conv_tracker[test_index[0]] = j
            print("convergence", (last_k_iter_sim == compare_k_iter_sim).all())
            break

    similar_pairs = list(itertools.combinations(similar_datasets, 2))
    similar_pairs_list = []
    for s in similar_pairs:
        similar_pairs_list.append(all_dataset_similarity[s[0], s[1]])
    selected_datasets_similarity.append(similar_pairs_list)
    print(np.mean(similar_pairs_list), np.std(similar_pairs_list))

    # actual wk to test by ground truth
    avg_dataset_similarity_header.append(
        all_dataset_similarity[test_index, similar_datasets])
    # regressor believe wk to test 
    reg_dataset_similarity_header.append(
        dataset_similarity_i[similar_datasets])
    # regressor believe wk to teston all datasets
    reg_dataset_similarity_all.append(dataset_similarity_i)

    # store ground truth
    dataset_header.append(datasets[data_ind[0]])
    selected_ap_header.append(ap_values[test_index, best_model_so_far][0])

    curr_models_test_ap_history.append(curr_models_test_ap_tracker)
    rank_tracker.append(298 - ap_ranks[test_index, best_model_so_far])

# save our result
result = np.column_stack((dataset_header, selected_ap_header))
result_df = pd.DataFrame(result, columns=['Dataset', 'ap'])
print("mean conv", np.mean(conv_tracker), "std conv", np.std(conv_tracker))
print('avg. rank', np.mean(rank_tracker))

# %%

# statistical tests

from scipy.stats import wilcoxon

# make sure the datasets are comoarable
sort_by_name = ap_full.sort_values('Dataset').to_numpy()[:, 1:]
random_baseline = np.mean(sort_by_name, axis=1)
sorted_ap_val = np.sort(sort_by_name, axis=1)
sorted_ap_val_1d = np.mean(sorted_ap_val, axis=0)
ap_name_sorted = ap_full.sort_values('Dataset').to_numpy()[:, 1:]
iforest_baseline = np.mean(ap_name_sorted[:, base_detectors_ranges[2]],
                           axis=1).astype(float)
our_result = result_df.sort_values('Dataset')['ap'].astype(float).tolist()

our_qth = 0
iforest_qth = 85
random_qth = 145

for i in range(n_models):
    print('top', i + 1,
          wilcoxon(sorted_ap_val[:, 296 - i], our_result,
                   alternative='greater'), np.mean(sorted_ap_val[:, 296 - i]),
          np.mean(our_result))
    if wilcoxon(sorted_ap_val[:, 296 - i], our_result, alternative='greater')[
        1] >= 0.05:
        our_qth = i + 1
        break

better_than_iforest = \
    wilcoxon(our_result, iforest_baseline, alternative='greater')[1] <= 0.05
better_than_iforest_p = np.round(
    wilcoxon(our_result, iforest_baseline, alternative='greater')[1],
    decimals=4)
better_than_random = \
    wilcoxon(our_result, random_baseline, alternative='greater')[1] <= 0.05
better_than_random_p = np.round(
    wilcoxon(our_result, random_baseline, alternative='greater')[1],
    decimals=4)

import numpy as np;

np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


colors = get_cmap(2)

#########11111111111111111111#####
df = pd.read_csv(os.path.join("intermediate_files", 'ap.csv'))
labels = ['ours', 'iforest_r', 'random']

df_np = df.to_numpy()
# df_list = df_np.T.tolist()
df_list = [our_result, iforest_baseline.tolist(), random_baseline.tolist()]

x = [1, 2, 3]

medianprops = dict(color="black", linewidth=1)
boxprops = dict(color="black", linewidth=1)

plt.figure(figsize=(6, 2))
box = plt.boxplot(df_list, labels=labels, patch_artist=True,
                  medianprops=medianprops, showfliers=True, widths=0.75,
                  vert=False)
plt.xticks(rotation=60, size=15)

for i in range(len(box['boxes'])):
    box['boxes'][i].set_facecolor('grey')

plt.text(0.75, 3.1, str(random_qth) + 'th', fontsize=10)
plt.text(0.75, 2.1, str(iforest_qth) + 'th', fontsize=10)
plt.text(0.75, 1.1, str(our_qth) + 'th', fontsize=10)

if better_than_iforest:
    plt.text(0.85, 2.1, 'p=' + str(better_than_iforest_p), fontsize=8,
             color='red')
else:
    plt.text(0.85, 2.1, 'p=' + str(better_than_iforest_p), fontsize=8,
             color='k')

if better_than_random:
    plt.text(0.85, 3.1, 'p=' + str(better_than_random_p), fontsize=8,
             color='red')
else:
    plt.text(0.85, 3.1, 'p=' + str(better_than_random_p), fontsize=8,
             color='k')

plt.yticks(size=12)
plt.xticks(np.arange(0, 1.1, 0.1), size=12)
plt.title('Test AP (median={med}, mean={mean}, std={std})'.format(
    med=np.round(np.median(our_result), decimals=2),
    mean=np.round(np.mean(our_result), decimals=2),
    std=np.round(np.std(our_result), decimals=2),
))
plt.tight_layout()
plt.show()
