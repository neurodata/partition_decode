import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss
from .dataset import generate_gaussian_parity
from .metrics import compute_df_gini_mean, compute_hellinger_dist
from tqdm import tqdm
import os
from datetime import datetime


def get_tree(
    method="rf", max_depth=1, n_estimators=1, max_leaf_nodes=None, random_state=1514
):
    """
    Initializes the decision forest and returns it

    Parameters
    ----------
    method: {"rf","gb"}, default="rf"
        The decision forest algorithm:
        "rf": random forest, "gb": gradient boosting

    max_depth: int or None, default=1
        The maximum depth of the tree.
        If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

    n_estimators: int, default=1
        The number of trees in the forest.

    max_leaf_nodes: int or None, default=None
        Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state: int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)


    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier or sklearn.ensemble.GradientBoostingClassifier
        A decision forest model.
    """
    if method == "gb":
        model = GradientBoostingClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=1,
            criterion="mse",
        )
    else:
        model = RandomForestClassifier(
            bootstrap=False,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
        )

    return model


def train_forest(
    method,
    xx,
    yy,
    true_posterior,
    exp_rep_i,
    train_n_samples=4096,
    test_n_samples=1000,
    max_node=None,
    n_est=10,
    exp_name="depth",
    verbose=False,
):
    if verbose:
        print("Experiment #: ", exp_rep_i)

    X_train, y_train = generate_gaussian_parity(
        n_samples=train_n_samples, angle_params=0
    )
    X_test, y_test = generate_gaussian_parity(n_samples=test_n_samples, angle_params=0)

    nodes, polytopes = [], []
    gini_score_train, gini_score_test = [], []
    train_error, test_error = [], []
    train_error_log, test_error_log = [], []
    hellinger_dist = []

    model = get_tree(method, max_depth=1)
    # for rep_i in range(reps):
    for depth in tqdm(range(1, max_node + n_est), position=0, leave=True):

        if depth < max_node:
            model.max_depth += 1
        else:
            model.n_estimators += 3
            model.max_depth += 15
            # model.warm_start=True

        model.fit(X_train, y_train)

        if method == "gb":
            nodes.append(
                sum([(estimator[0].get_n_leaves()) for estimator in model.estimators_])
            )
        else:
            nodes.append(
                sum([estimator.get_n_leaves() for estimator in model.estimators_])
            )
        leaf_idxs = model.apply(X_train)
        polytopes.append(len(np.unique(leaf_idxs, axis=0)))
        gini_score_train.append(compute_df_gini_mean(model, X_train, y_train))
        gini_score_test.append(compute_df_gini_mean(model, X_test, y_test))
        train_error.append(1 - model.score(X_train, y_train))
        test_error.append(1 - model.score(X_test, y_test))
        train_error_log.append(log_loss(y_train, model.predict(X_train)))
        test_error_log.append(log_loss(y_test, model.predict(X_test)))
        # ece_error.append( brier_score_loss(y_train, model.predict(X_train)))
        rf_posteriors_grid = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        hellinger_dist.append(
            compute_hellinger_dist(rf_posteriors_grid, true_posterior)
        )
    nodes = np.array(nodes)
    polytopes = np.array(polytopes)
    train_error = np.array(train_error)
    test_error = np.array(test_error)
    train_error_log = np.array(train_error_log)
    test_error_log = np.array(test_error_log)
    gini_score_train = np.array(gini_score_train)
    gini_score_test = np.array(gini_score_test)
    hellinger_dist = np.array(hellinger_dist)

    path = "../results/df/" + exp_name + "/"
    datetime_now = datetime.now(tz=None)
    datetime_str = datetime_now.strftime("%m_%d_%Y_%H_%M_%S")
    os.makedirs(path, exist_ok=True)

    np.save(
        path + str(exp_rep_i) + "_" + datetime_str + ".npy",
        [
            nodes,
            polytopes,
            train_error,
            test_error,
            train_error_log,
            test_error_log,
            gini_score_train,
            gini_score_test,
            hellinger_dist,
        ],
    )

    return [
        nodes,
        polytopes,
        train_error,
        test_error,
        train_error_log,
        test_error_log,
        gini_score_train,
        gini_score_test,
        hellinger_dist,
    ]
