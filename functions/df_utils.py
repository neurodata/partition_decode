import numpy as np

from joblib import Parallel, delayed
import multiprocessing

from .dataset import generate_gaussian_parity
from .forest import train_forest, get_tree
from .metrics import compute_true_posterior
import os


def run_df_experiments(
    train_n_samples=4096,
    test_n_samples=1000,
    n_reps=100,
    max_node=None,
    n_est=10,
    exp_alias="deep",
):

    grid_xx, grid_yy = np.meshgrid(np.arange(-2, 2, 4 / 100), np.arange(-2, 2, 4 / 100))
    grid_true_posterior = np.array(
        [compute_true_posterior(x) for x in (np.c_[grid_xx.ravel(), grid_yy.ravel()])]
    )

    train_mean_error, test_mean_error = [], []
    train_mean_error_log, test_mean_error_log = [], []
    gini_train_mean_score, gini_test_mean_score = [], []

    X_train, y_train = generate_gaussian_parity(
        n_samples=train_n_samples, angle_params=0
    )
    # X_test, y_test = generate_gaussian_parity(n_samples=test_n_samples, angle_params=0)

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

    train_error, test_error = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    train_error_log, test_error_log = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    gini_score_train, gini_score_test = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    hellinger_dist = [list() for _ in range(n_reps)]
    nodes = [list() for _ in range(n_reps)]
    polys = [list() for _ in range(n_reps)]
    # for depth in tqdm(range(1, max_node + n_est), position=0, leave=True):
    # for rep_i in tqdm(range(n_reps), position=0, leave=True):
    def one_run(rep_i):
        [
            nodes[rep_i],
            polys[rep_i],
            train_error[rep_i],
            test_error[rep_i],
            train_error_log[rep_i],
            test_error_log[rep_i],
            gini_score_train[rep_i],
            gini_score_test[rep_i],
            hellinger_dist[rep_i],
        ] = train_forest(
            method,
            grid_xx,
            grid_yy,
            grid_true_posterior,
            rep_i,
            train_n_samples=train_n_samples,
            test_n_samples=test_n_samples,
            max_node=max_node,
            n_est=n_est,
            exp_name=exp_alias,
        )

    n_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=-1)(delayed(one_run)(i) for i in range(n_reps))

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


def read_df_results(n_reps, exp_alias="depth"):

    dir_path = "../results/df/" + exp_alias + "/"
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".npy"):
                file_paths.append(os.path.join(root, file))

    result = lambda: None
    train_error, test_error = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    train_error_log, test_error_log = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    gini_score_train, gini_score_test = [list() for _ in range(n_reps)], [
        list() for _ in range(n_reps)
    ]
    nodes = [list() for _ in range(n_reps)]
    polytopes = [list() for _ in range(n_reps)]
    hellinger_dist = [list() for _ in range(n_reps)]

    for rep_i in range(n_reps):
        [
            nodes[rep_i],
            polytopes[rep_i],
            train_error[rep_i],
            test_error[rep_i],
            train_error_log[rep_i],
            test_error_log[rep_i],
            gini_score_train[rep_i],
            gini_score_test[rep_i],
            hellinger_dist[rep_i],
        ] = np.load(
            file_paths[rep_i],
        )
        # np.load("../results/xor_rf_dd_" + exp_alias + "_" + str(rep_i) + ".npy")

    result.train_err_list = np.array(train_error)  # .mean(axis=0)
    result.test_err_list = np.array(test_error)  # .mean(axis=0)
    result.train_err_log_list = np.array(train_error_log)  # .mean(axis=0)
    result.test_error_log_list = np.array(test_error_log)  # .mean(axis=0)
    result.n_nodes = np.array(nodes).mean(axis=0)
    result.n_polytopes_list = np.array(polytopes)  # .mean(axis=0)
    result.gini_train_list = np.array(gini_score_train)  # .mean(axis=0)
    result.gini_test_list = np.array(gini_score_test)  # .mean(axis=0)
    result.hellinger_dist_list = np.array(hellinger_dist)  # .mean(axis=0)

    return result


"""
  Example to run the `Deep RF` vs `Shallow RF` experiments
  and plot the figure.
"""
### via CLI

# import argparse

# parser = argparse.ArgumentParser(description='Run a double descent experiment.')

# parser.add_argument('--deep', action="store_true", default=False)
# parser.add_argument('-n_reps', action="store", dest="n_reps", type=int)
# parser.add_argument('-n_est', action="store", dest="n_est", type=int)
# parser.add_argument('-max_node', action="store", dest="max_node", default=None, type=int)
# parser.add_argument('-cov_scale', action="store", dest="cov_scale", default=1.0, type=float)


# args = parser.parse_args()

# exp_alias = "deep" if args.deep else "shallow"

# result = rf_dd_exp(max_node=args.max_node, n_est=args.n_est, n_reps=args.n_reps, exp_alias = exp_alias)

# --------------

### on a notebook
# n_reps = 1  # the number of repetitions of a single run of the algorithm
# # Run DeepRF
# error_deep = run_df_experiment(max_node=None, n_est=10, n_reps=n_reps, exp_alias="deep")
# # Run ShallowRF
# error_shallow = run_df_experiment(max_node=15, n_est=100, n_reps=n_reps, exp_alias="shallow")
# # np.save('errors.npy', [error_5, error_dd])

# error_deep = read_df_results(n_reps, exp_alias="deep")
# error_shallow = read_df_results(n_reps, exp_alias="shallow")

# results = [error_deep, error_shallow]
# titles = ["RF with overfitting trees", "RF with shallow trees"]

# from .plots import plot_df_results
# plot_df_results(results, titles)
