from typing import Dict
import pandas as pd
import numpy as np
import sklearn.metrics as skm


def get_fpr_tpr_thr(y_true, y_pred, pos_label):
    """Computes the FPR, TPR and THRESHOLD for a binary classification problem.
        * `y_score >= threhold` is classified as `pos_label`.
    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        pos_label ([type]): [description]
    Returns:
        fpr : Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= `thresholds[i]`.
        tpr : Increasing true positive rates such that element `i` is the true
            positive rate of predictions with score >= `thresholds[i]`.
        thresholds : Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.
    """
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, pos_label=pos_label)
    return fpr, tpr, thresholds


def get_precision_recall_thr(y_true, y_pred, pos_label=1):
    precision, recall, thresholds = skm.precision_recall_curve(
        y_true, y_pred, pos_label=pos_label
    )
    return precision, recall, thresholds

def compute_auroc(fpr, tpr):
    return skm.auc(fpr, tpr)

def compute_fpr_tpr_thr_given_tpr_level(fpr, tpr, thresholds, tpr_level):
    if all(tpr < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    # elif all(tpr >= tpr_level):
    #     # All thresholds allow TPR >= tpr level, so find lowest possible FPR
    #     idx = np.argmin(fpr)
    else:
        idxs = [i for i, x in enumerate(tpr) if x >= tpr_level]
        idx = min(idxs)
    return fpr[idx], tpr[idx], thresholds[idx]

def compute_fpr_tpr(fpr, tpr, thresholds):
    idx = min(i for i, x in enumerate(tpr) if x >= 0.95)
    return fpr[idx], tpr[idx], thresholds[idx]

def compute_metrics(df_stats, category: str, metrics: list):
    eval_metrics = []
    for metric in metrics:
        detector_scores = df_stats[metric].values
        fpr, tpr, thresholds = get_fpr_tpr_thr(df_stats[category].values, detector_scores, pos_label=1)
        fpr_at_90tpr, _, _ = compute_fpr_tpr_thr_given_tpr_level(fpr, tpr, thresholds, tpr_level=0.9)
        precision, recall, thresholds = get_precision_recall_thr(df_stats[category].values, detector_scores, pos_label=1)
        auc_roc = compute_auroc(fpr, tpr)
        auc = {"metric": metric, "auc-ROC": auc_roc * 100, "fprat90tpr": fpr_at_90tpr * 100}
        eval_metrics.append(auc)
    df = pd.DataFrame(eval_metrics)
    return df.sort_values(by="auc-ROC", ascending = False)