from typing import List
import pandas as pd
import numpy as np
from metrics import *
import ot
from tqdm import tqdm

def wasserstein_distance_against_uni(u, v, metric_space):
    if metric_space == "uni":
        wass_dist = np.linalg.norm(u-v, ord=1) / 2
    else:
        size = max(u.size, v.size)
        u_vec = np.pad(u, (0, size - u.size), 'constant', constant_values=0)
        v_vec = np.pad(v, (0, size - v.size), 'constant', constant_values=0)
        wass_dist = np.linalg.norm(np.dot(np.tril(np.ones((size, size))), u_vec-v_vec), ord=1)
        # M = ot.dist(x1 = np.expand_dims(range(u.shape[0]), axis=1), x2 = np.expand_dims(range(v.shape[0]), axis=1), metric="minkowski", p=1)
        # wass_dist = ot.emd(
        #                 a=u,
        #                 b=v,
        #                 M=M,
        #                 log=True,
        #             )[1]["cost"]
    return wass_dist

def wasserstein_distance(u, df, metric_space):
    attn = df.attn.values # make sure to have the weights in the "attn" column
    wass_dists = []
    for i in range(len(attn)):
        v = attn[i].sum(1) / attn[i].shape[1] # marginalize over target tokens to obtain the attention mass on source tokens
        size = max(u.size, v.size)
        u_vec = np.pad(u, (0, size - u.size), 'constant', constant_values=0)
        v_vec = np.pad(v, (0, size - v.size), 'constant', constant_values=0)
        if metric_space == "uni": # 0/1 cost function
            wass_dist = np.linalg.norm(u_vec-v_vec, ord=1) / 2 # equivalent to total variation distance
        else:
            wass_dist = np.linalg.norm(np.dot(np.tril(np.ones((size, size))), u_vec-v_vec), ord=1) # earth mover's distance
            # Alternative option: use OT library
            # M = ot.dist(x1 = np.expand_dims(range(u.shape[0]), axis=1), x2 = np.expand_dims(range(v.shape[0]), axis=1), metric="minkowski", p=1)
            # wass_dist = ot.emd(
            #                 a=u,
            #                 b=v,
            #                 M=M,
            #                 log=True,
            #             )[1]["cost"]
        wass_dists.append(wass_dist)
    return wass_dists

def apply_length_restrictions(u_matrix, dataset_attn_train, args):
    no_samples_wass_dist = args.no_samples
    seed = args.seed
    length_window = args.length_window

    _, mt_len = u_matrix.shape

    if length_window != 0.0:
        dataset_mt_window_constraint = dataset_attn_train.loc[
            (
                dataset_attn_train["mt_len"]
                < (1 + length_window / 100) * mt_len
            )
            & (
                dataset_attn_train["mt_len"]
                > (1 - length_window / 100) * mt_len
            )
        ]
        remaining_samples = no_samples_wass_dist - len(dataset_mt_window_constraint)
        if len(dataset_mt_window_constraint) >= no_samples_wass_dist:
            dataset_constraint = dataset_mt_window_constraint.sample(
                no_samples_wass_dist, random_state=seed
            )
        else:
            dataset_constraint = dataset_mt_window_constraint
            if len(dataset_constraint) == 0:
                dataset_constraint = dataset_attn_train.sample(
                no_samples_wass_dist, random_state=seed
                )
    else:
        dataset_constraint = dataset_attn_train.sample(no_samples_wass_dist, random_state=seed)

    return dataset_constraint

def compute_detection_metrics(dataset_stats, args, bottom_k=4, metrics=["wass_to_data"]):
    df_all = compute_metrics(dataset_stats, category="is_hall", metrics = metrics)
    auroc_all = df_all["auc-ROC"].values[0]
    fprat90_all = df_all["fprat90tpr"].values[0]

    dataset_stats_osc = dataset_stats.loc[(dataset_stats.is_osc == 1) | (dataset_stats.is_hall == 0)]
    df_osc = compute_metrics(dataset_stats_osc, category="is_osc", metrics = metrics)
    auroc_osc = df_osc["auc-ROC"].values[0]
    fprat90_osc = df_osc["fprat90tpr"].values[0]

    dataset_stats_fd = dataset_stats.loc[(dataset_stats.is_fd == 1) | (dataset_stats.is_hall == 0)]
    df_fd = compute_metrics(dataset_stats_fd, category="is_fd", metrics = metrics)
    auroc_fd = df_fd["auc-ROC"].values[0]
    fprat90_fd = df_fd["fprat90tpr"].values[0]

    dataset_stats_sd = dataset_stats.loc[(dataset_stats.is_sd == 1) | ((dataset_stats.is_hall == 0))]
    df_sd = compute_metrics(dataset_stats_sd, category="is_sd", metrics = metrics)
    auroc_sd = df_sd["auc-ROC"].values[0]
    fprat90_sd = df_sd["fprat90tpr"].values[0]

    auroc_metrics = {"auroc": {"all": auroc_all, "osc": auroc_osc, "fd": auroc_fd, "sd": auroc_sd},\
                    "fprat90tpr": {"all": fprat90_all, "osc": fprat90_osc, "fd": fprat90_fd, "sd": fprat90_sd}}
    
    return auroc_metrics

def wass_dist_computation(dataset_stats, dataset_attn_train, args):
    wasserstein_distances = []
    for idx in tqdm(dataset_stats.index):
        attn_matrix = dataset_stats.loc[idx].attn # (target_len, source_len); make sure to have the weights in the "attn" column
        attn_dist = attn_matrix.sum(1) / attn_matrix.shape[1] # marginalize over target tokens to obtain the attention mass on source tokens
        if args.compare_to_uni:
            uni_dist = ot.unif(attn_dist.shape[0])  # uniform distribution
            wasserstein_dist = wasserstein_distance_against_uni(u=attn_dist, v=uni_dist, metric_space=args.metric_space) # compute wasserstein distance
            wasserstein_distances.append(wasserstein_dist)
        else:
            dataset_constraint = apply_length_restrictions(u_matrix=attn_matrix, dataset_attn_train=dataset_attn_train, args=args) # apply length restrictions
            wasserstein_distances_sample = wasserstein_distance(u=attn_dist, df=dataset_constraint, metric_space=args.metric_space) # compute wasserstein distance
            wasserstein_distances.append(wasserstein_distances_sample)
    return wasserstein_distances

def convert_score(X, scores_old_min, scores_old_max, scores_new_min=0, scores_new_max=1):      
    X = scores_new_min + ((X - scores_new_min) * (scores_new_max - scores_new_min) / (scores_old_max - scores_old_min)) 
    return X  