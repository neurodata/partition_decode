""" Metrics to assess performance of the models as training proceeds.
"""

import numpy as np
import torch
from torch import nn

from sklearn.metrics import log_loss, brier_score_loss


"""
Common metrics
"""

# Gini impurity
def gini_impurity(P1=0, P2=0):
    denom = P1 + P2
    Ginx = 2 * (P1 / denom) * (P2 / denom)
    return Ginx


# Hellinger distance
def hellinger(p, q):
    """Hellinger distance between two discrete distributions.
    In pure Python. Original.
    """
    return sum(
        [
            (np.sqrt(t[0]) - np.sqrt(t[1])) * (np.sqrt(t[0]) - np.sqrt(t[1]))
            for t in zip(p, q)
        ]
    ) / np.sqrt(2.0)


def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
    Same as original version but without list comprehension.
    """
    return np.mean(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2))


def compute_hellinger_dist(p, q):
    """Hellinger distance between two discrete distributions.
    For Python >= 3.5 only"""
    return np.mean(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2))
    # z = np.sqrt(p) - np.sqrt(q)
    # return np.sqrt(z @ z / 2)


def compute_true_posterior(x, means=None):
    """Computes the true posterior of the Gaussian XOR"""

    if means is None:
        means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]

    mu01, mu02, mu11, mu12 = means  # [[-1, -1], [1, 1], [-1, 1], [1, -1]]

    cov = 1 * np.eye(2)
    inv_cov = np.linalg.inv(cov)

    p0 = (
        np.exp(-(x - mu01) @ inv_cov @ (x - mu01).T)
        + np.exp(-(x - mu02) @ inv_cov @ (x - mu02).T)
    ) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

    p1 = (
        np.exp(-(x - mu11) @ inv_cov @ (x - mu11).T)
        + np.exp(-(x - mu12) @ inv_cov @ (x - mu12).T)
    ) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

    return [p1 / (p0 + p1), p0 / (p0 + p1)]


## ECE loss
def bin_data(y, n_bins):
    """
    Partitions the data into ordered bins based on
    the probabilities. Returns the binned indices.
    """
    edges = np.linspace(0, 1, n_bins)
    bin_idx = np.digitize(y, edges, right=True)
    binned_idx = [np.where(bin_idx == i)[0] for i in range(n_bins)]

    return binned_idx


def bin_stats(y_true, y_proba, bin_idx):
    # mean accuracy within each bin
    bin_acc = [
        np.equal(np.argmax(y_proba[idx], axis=1), y_true[idx]).mean()
        if len(idx) > 0
        else 0
        for idx in bin_idx
    ]
    # mean confidence of prediction within each bin
    bin_conf = [
        np.mean(np.max(y_proba[idx], axis=1)) if len(idx) > 0 else 0 for idx in bin_idx
    ]

    return np.asarray(bin_acc), np.asarray(bin_conf)


def compute_ece_loss(y_true, y_proba, n_bins=10):
    """Computes the ECE loss"""
    bin_idx = bin_data(y_proba.max(axis=1), n_bins)
    n = len(y_true)

    bin_acc, bin_conf = bin_stats(y_true, y_proba, bin_idx)
    bin_sizes = [len(idx) for idx in bin_idx]

    ece_loss = np.sum(np.abs(bin_acc - bin_conf) * np.asarray(bin_sizes)) / n

    return ece_loss


"""
Deep Net metrics
"""

# Average stability
def compute_avg_stability(model, hybrid_set):
    """
    Computes the average stability of a model
    based on https://mlstory.org/generalization.html#algorithmic-stability
    """
    stab_dif = 0
    N = len(hybrid_set)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for i in range(N):
        model_hybrid = copy.deepcopy(model)

        ghost_loss = loss_func(model(hybrid_set[i][0]), hybrid_set[i][1])
        loss = train_model(model_hybrid, hybrid_set[i][0], hybrid_set[i][1])
        stab_dif += ghost_loss.detach().cpu().numpy().item() - loss[-1]

    return stab_dif / N


def compute_gini_mean(polytope_memberships, predicts):
    """
    Compute the mean Gini impurity based on
    the polytope membership of the points and
    the model prediction of the labels.
    """
    gini_mean_score = []

    for l in np.unique(polytope_memberships):

        cur_l_idx = predicts[polytope_memberships == l]
        pos_count = np.sum(cur_l_idx)
        neg_count = len(cur_l_idx) - pos_count
        gini = gini_impurity(pos_count, neg_count)
        gini_mean_score.append(gini)

    return np.array(gini_mean_score).mean()


def get_gini_list(polytope_memberships, predicts):
    """
    Computes the Gini impurity same as compute_gini_mean
    but returns the whole list
    """
    gini_score = np.zeros(polytope_memberships.shape)

    for l in np.unique(polytope_memberships):
        idx = np.where(polytope_memberships == l)[0]
        cur_l_idx = predicts[polytope_memberships == l]
        pos_count = np.sum(cur_l_idx)
        neg_count = len(cur_l_idx) - pos_count
        gini = gini_impurity(pos_count, neg_count)
        gini_score[idx] = gini

    return np.array(gini_score)


"""
Decision Forest metrics
"""


def compute_df_gini_mean(model, data, label):
    leaf_idxs = model.apply(data)
    predict = label
    gini_mean_score = []
    for t in range(leaf_idxs.shape[1]):
        gini_arr = []
        for l in np.unique(leaf_idxs[:, t]):
            cur_l_idx = predict[leaf_idxs[:, t] == l]
            pos_count = np.sum(cur_l_idx)
            neg_count = len(cur_l_idx) - pos_count
            gini = gini_impurity(pos_count, neg_count)
            gini_arr.append(gini)

        gini_mean_score.append(np.array(gini_arr).mean())
    return np.array(gini_mean_score).mean()
