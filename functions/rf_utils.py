import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn import metrics
from sklearn import tree
from tqdm import tqdm

from sklearn.metrics import log_loss, brier_score_loss
from joblib import Parallel, delayed
import multiprocessing

from utils import ece, generate_gaussian_parity
import argparse


def generate_gaussian_parity(n, cov_scale=1, angle_params=None, k=1, acorn=None):
    #     means = [[-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [-1.5, 1.5]]
    means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    blob = np.concatenate(
        [
            np.random.multivariate_normal(
                mean, cov_scale * np.eye(len(mean)), size=int(n / 4)
            )
            for mean in means
        ]
    )

    X = np.zeros_like(blob)
    Y = np.concatenate([np.ones((int(n / 4))) * int(i < 2) for i in range(len(means))])
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(
        angle_params * np.pi / 180
    )
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(
        angle_params * np.pi / 180
    )
    return X, Y.astype(int)


def get_tree(method="rf", max_depth=1, n_estimators=1, max_leaf_nodes=None):
    if method == "gb":
        rf = GradientBoostingClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=1514,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=1,
            criterion="mse",
        )
    else:
        rf = RandomForestClassifier(
            bootstrap=False,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=1514,
            max_leaf_nodes=max_leaf_nodes,
        )

    return rf


def gini_impurity(P1=0, P2=0):
    denom = P1 + P2
    Ginx = 2 * (P1 / denom) * (P2 / denom)
    return Ginx


def gini_impurity_mean(rf, data, label):
    leaf_idxs = rf.apply(data)
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


def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
    Same as original version but without list comprehension
    """
    return np.mean(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2))


def pdf(x):
    mu01, mu02, mu11, mu12 = [[-1, -1], [1, 1], [-1, 1], [1, -1]]

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


def rf_dd_exp(N=4096, reps=100, max_node=None, n_est=10, exp_alias="depth"):

    xx, yy = np.meshgrid(np.arange(-2, 2, 4 / 100), np.arange(-2, 2, 4 / 100))
    true_posterior = np.array([pdf(x) for x in (np.c_[xx.ravel(), yy.ravel()])])

    train_mean_error, test_mean_error = [], []
    train_mean_error_log, test_mean_error_log = [], []
    gini_train_mean_score, gini_test_mean_score = [], []

    # X, y = get_sample(N)
    X_train, y_train = generate_gaussian_parity(n=N, angle_params=0)
    X_test, y_test = generate_gaussian_parity(n=N, angle_params=0)

    method = "rf"

    if max_node is None:
        rf = get_tree(method, max_depth=None)
        rf.fit(X_train, y_train)
        if method == "gb":
            max_node = (
                sum([estimator[0].get_n_leaves() for estimator in rf.estimators_])
            ) + 50
        else:
            max_node = (
                sum([estimator.get_n_leaves() for estimator in rf.estimators_]) + 50
            )

    train_error, test_error = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    train_error_log, test_error_log = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    gini_score_train, gini_score_test = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    ece_error = [list() for _ in range(reps)]
    nodes = [list() for _ in range(reps)]
    polys = [list() for _ in range(reps)]
    # for depth in tqdm(range(1, max_node + n_est), position=0, leave=True):
    # for rep_i in tqdm(range(reps), position=0, leave=True):
    def one_run(rep_i):
        print(rep_i)
        X_train, y_train = generate_gaussian_parity(n=N, angle_params=0)
        X_test, y_test = generate_gaussian_parity(n=1000, angle_params=0)

        rf = get_tree(method, max_depth=1)
        # for rep_i in range(reps):
        for depth in tqdm(range(1, max_node + n_est), position=0, leave=True):

            if depth < max_node:
                rf.max_depth += 1
            else:
                rf.n_estimators += 3
                rf.max_depth += 15
                # rf.warm_start=True

            rf.fit(X_train, y_train)

            if method == "gb":
                nodes[rep_i].append(
                    sum([(estimator[0].get_n_leaves()) for estimator in rf.estimators_])
                )
            else:
                nodes[rep_i].append(
                    sum([estimator.get_n_leaves() for estimator in rf.estimators_])
                )
            leaf_idxs = rf.apply(X_train)
            polys[rep_i].append(len(np.unique(leaf_idxs)))
            gini_score_train[rep_i].append(gini_impurity_mean(rf, X_train, y_train))
            gini_score_test[rep_i].append(gini_impurity_mean(rf, X_test, y_test))
            train_error[rep_i].append(1 - rf.score(X_train, y_train))
            test_error[rep_i].append(1 - rf.score(X_test, y_test))
            train_error_log[rep_i].append(log_loss(y_train, rf.predict(X_train)))
            test_error_log[rep_i].append(log_loss(y_test, rf.predict(X_test)))
            #             ece_error[rep_i].append( brier_score_loss(y_train, rf.predict(X_train)))
            rf_posteriors_grid = rf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            ece_error[rep_i].append(
                hellinger_explicit(rf_posteriors_grid, true_posterior)
            )
        nodes[rep_i] = np.array(nodes[rep_i])
        polys[rep_i] = np.array(polys[rep_i])
        train_error[rep_i] = np.array(train_error[rep_i])
        test_error[rep_i] = np.array(test_error[rep_i])
        train_error_log[rep_i] = np.array(train_error_log[rep_i])
        test_error_log[rep_i] = np.array(test_error_log[rep_i])
        gini_score_train[rep_i] = np.array(gini_score_train[rep_i])
        gini_score_test[rep_i] = np.array(gini_score_test[rep_i])
        ece_error[rep_i] = np.array(ece_error[rep_i])

        np.save(
            "results/xor_rf_dd_" + exp_alias + "_" + str(rep_i) + ".npy",
            [
                nodes[rep_i],
                polys[rep_i],
                train_error[rep_i],
                test_error[rep_i],
                train_error_log[rep_i],
                test_error_log[rep_i],
                gini_score_train[rep_i],
                gini_score_test[rep_i],
                ece_error[rep_i],
            ],
        )

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=-1)(delayed(one_run)(i) for i in range(reps))

    train_mean_error = np.array(train_error).mean(axis=0)
    test_mean_error = np.array(test_error).mean(axis=0)
    train_mean_error_log = np.array(train_error_log).mean(axis=0)
    test_mean_error_log = np.array(test_error_log).mean(axis=0)
    nodes_mean = np.array(nodes).mean(axis=0)
    gini_train_mean_score = np.array(gini_score_train).mean(axis=0)
    gini_test_mean_score = np.array(gini_score_test).mean(axis=0)
    error_dict = {
        "train_err": train_mean_error,
        "test_err": test_mean_error,
        "train_err_log": train_mean_error_log,
        "test_err_log": test_mean_error_log,
        "train_gini": gini_train_mean_score,
        "test_gini": gini_test_mean_score,
        "nodes": nodes_mean,
    }
    return error_dict


def read_results(reps, exp_alias="depth"):
    train_error, test_error = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    train_error_log, test_error_log = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    gini_score_train, gini_score_test = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    nodes = [list() for _ in range(reps)]
    polys = [list() for _ in range(reps)]
    ece_error = [list() for _ in range(reps)]

    for rep_i in range(reps):
        [
            nodes[rep_i],
            polys[rep_i],
            train_error[rep_i],
            test_error[rep_i],
            train_error_log[rep_i],
            test_error_log[rep_i],
            gini_score_train[rep_i],
            gini_score_test[rep_i],
            ece_error[rep_i],
        ] = np.load("results/xor_rf_dd_" + exp_alias + "_" + str(rep_i) + ".npy")

    train_mean_error = np.array(train_error).mean(axis=0)
    test_mean_error = np.array(test_error).mean(axis=0)
    train_mean_error_log = np.array(train_error_log).mean(axis=0)
    test_mean_error_log = np.array(test_error_log).mean(axis=0)
    nodes_mean = np.array(nodes).mean(axis=0)
    polys_mean = np.array(polys).mean(axis=0)
    gini_train_mean_score = np.array(gini_score_train).mean(axis=0)
    gini_test_mean_score = np.array(gini_score_test).mean(axis=0)
    ece_mean_error = np.array(ece_error).mean(axis=0)
    error_dict = {
        "train_err": train_mean_error,
        "test_err": test_mean_error,
        "train_err_log": train_mean_error_log,
        "test_err_log": test_mean_error_log,
        "train_gini": gini_train_mean_score,
        "test_gini": gini_test_mean_score,
        " ece_error": ece_mean_error,
        "polys": polys_mean,
        "nodes": nodes_mean,
    }

    return error_dict


# parser = argparse.ArgumentParser(description='Run a double descent experiment.')

# parser.add_argument('--depth', action="store_true", default=False)
# parser.add_argument('-reps', action="store", dest="reps", type=int)
# parser.add_argument('-n_est', action="store", dest="n_est", type=int)
# parser.add_argument('-max_node', action="store", dest="max_node", default=None, type=int)
# parser.add_argument('-cov_scale', action="store", dest="cov_scale", default=1.0, type=float)


# args = parser.parse_args()

# exp_alias = "depth" if args.depth else "width"

# error_dd = rf_dd_exp(max_node=args.max_node, n_est=args.n_est, reps=args.reps, exp_alias = exp_alias)
