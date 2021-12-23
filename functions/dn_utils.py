import numpy as np
import torch
from torch import nn

import copy
import os

from .dataset import *
from .network import *
from .plots import *
from .metrics import compute_true_posterior, compute_gini_mean, compute_hellinger_dist


def run_dn_experiment(
    n_iterations,
    increase_depth=False,
    increase_width=False,
    n_reps=100,
    width=3,
    cov_scale=1,
    verbose=False,
):
    """
    Main function to run the `Increasing Depth` and `Increasing Width` experiments

    Parameters
    ----------

    n_iterations: int
        The number of iterations to keep increasing the depth or width in a single experiment.

    increase_depth: bool, default=False
        A boolean indicating if it is the increasing depth experiment.

    increase_width: bool, default=False
        A boolean indicating if it is the increase width experiment.

    n_reps : int
        The number of repetitions of the experiment.

    width: int, default=3
        The initial width of the hidden layers.

    verbose: bool, default = False
        A boolean indicating the production of detailed logging information during training.

    Returns
    -------
    results : ndarray
        An array that consists the result for the `Increasing Width` and `Increasing Depth` experiments.
        A result object in the results array should have the following attributes:
            result.n_pars, result.train_err_list, result.test_err_list, result.train_loss_list,
            result.test_loss_list, result.gini_train, result.gini_test, result.n_polytopes_list
    """

    assert (
        increase_depth != increase_width
    ), "Only one of `increase_depth` or `increase_width` should be true!"

    result = lambda: None

    xx, yy = np.meshgrid(np.arange(-2, 2, 4 / 100), np.arange(-2, 2, 4 / 100))
    true_posterior = np.array(
        [compute_true_posterior(x) for x in (np.c_[xx.ravel(), yy.ravel()])]
    )

    rep_full_list = []

    train_x, train_y, test_x, test_y, hybrid_sets = get_dataset(
        n_samples=1000, cov_scale=cov_scale, include_hybrid=True
    )

    penultimate_vars_reps = []

    for rep in range(n_reps):  # 25
        if verbose:
            print("repetition #" + str(rep))

        ## Shuffle train set labels for activation variation panel
        # train_y_tmp = torch.clone(train_y)
        # train_y[train_y_tmp==0] = 1
        # train_y[train_y_tmp==1] = 0
        # test_y_tmp = torch.clone(test_y)
        # test_y[test_y_tmp==0] = 1
        # test_y[test_y_tmp==1] = 0

        losses_list = []
        n_pars = []
        n_polytopes_list = []
        hellinger_list = []
        gini_train, gini_test = [], []

        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []

        penultimate_acts = []
        penultimate_nodes = []
        penultimate_err = []
        penultimate_poly = []
        penultimate_vars = []

        avg_stab_list = []
        bias_list = []
        var_list = []

        for i in range(1, n_iterations):
            if verbose:
                print("iteration #", i)

            ## Increasing Depth
            if increase_depth:
                if i < 5:
                    model = get_model(n_hidden=i, hidden_size=i, bn=False)
                else:
                    model = get_model(n_hidden=i, bn=False)
            else:
                ## Increasing Width
                model = get_model(hidden_size=i, n_hidden=width, bn=False)

            n_par = sum(p.numel() for p in model.parameters())

            losses = train_model(model, train_x, train_y)

            polytopes, penultimate_act = get_polytopes(
                model, train_x, penultimate=False
            )
            n_polytopes = len(np.unique(polytopes[0]))

            if increase_depth:
                n_nodes = i * 20 if i > 5 else i * i
            else:
                n_nodes = i * 3

            penultimate_acts.append(penultimate_act)
            penultimate_vars.append(list(np.var(penultimate_act, axis=0)))

            with torch.no_grad():
                pred_train, pred_test = model(train_x), model(test_x)

                gini_impurity_train = compute_gini_mean(
                    polytopes[0], torch.sigmoid(pred_train).round().cpu().data.numpy()
                )
                polytopes_test, _ = get_polytopes(model, test_x, penultimate=False)
                gini_impurity_test = compute_gini_mean(
                    polytopes_test[0],
                    torch.sigmoid(pred_test).round().cpu().data.numpy(),
                )

                rf_posteriors_grid = model(
                    torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).cuda()
                )
                class_1_posteriors = (
                    torch.sigmoid(rf_posteriors_grid).detach().cpu().numpy()
                )
                pred_proba = np.concatenate(
                    [1 - class_1_posteriors, class_1_posteriors], axis=1
                )

                hellinger_loss = compute_hellinger_dist(pred_proba, true_posterior)

                train_y = train_y.type_as(pred_train)
                test_y = test_y.type_as(pred_test)
                train_loss = torch.nn.BCEWithLogitsLoss()(pred_train, train_y)
                # train_acc = (torch.argmax(pred_train,1) == torch.argmax(train_y,1)).sum().cpu().data.numpy().item() / train_y.size(0)
                train_acc = (
                    torch.sigmoid(pred_train).round() == train_y
                ).sum().cpu().data.numpy().item() / train_y.size(0)
                test_loss = torch.nn.BCEWithLogitsLoss()(pred_test, test_y)
                # test_acc = (torch.argmax(pred_test,1) == torch.argmax(test_y,1)).sum().cpu().data.numpy().item() / test_y.size(0)
                test_acc = (
                    torch.sigmoid(pred_test).round() == test_y
                ).sum().cpu().data.numpy().item() / test_y.size(0)

            ## Uncomment to plot the decision boundaries
            # plot_decision_boundaries(model, n_nodes, n_polytopes, 1-test_acc, plot_type='all', plot_name="depth" if increase_depth else "width")

            losses_list.append(losses)
            n_pars.append(n_par)
            n_polytopes_list.append(n_polytopes)

            train_loss_list.append(train_loss.item())
            test_loss_list.append(test_loss.item())
            train_acc_list.append(1 - train_acc)
            test_acc_list.append(1 - test_acc)
            hellinger_list.append(hellinger_loss)
            gini_train.append(gini_impurity_train)
            gini_test.append(gini_impurity_test)

            avg_stab = 0  # compute_avg_stability(model, hybrid_sets)
            bias, var = 0, 0  # compute_bias_variance(model, test_x, test_y, T=100)
            avg_stab_list.append(avg_stab)
            bias_list.append(bias)
            var_list.append(var)

        rep_full_list.append(
            [
                losses_list,
                train_loss_list,
                test_loss_list,
                train_acc_list,
                test_acc_list,
                hellinger_list,
                n_polytopes_list,
                gini_train,
                gini_test,
                avg_stab_list,
                bias_list,
                var_list,
            ]
        )
        penultimate_vars_reps.append(penultimate_vars)

    result.n_pars = n_pars
    [
        result.full_loss_list,
        result.train_loss_list,
        result.test_loss_list,
        result.test_err_list,
        result.train_err_list,
        result.hellinger_list,
        result.n_polytopes_list,
        result.gini_train,
        result.gini_test,
        result.avg_stab,
        result.bias,
        result.var,
    ] = extract_losses(rep_full_list)

    result.penultimate_vars_reps = penultimate_vars_reps

    return result


# Losses
def extract_losses(rep_full_list):
    """
    Extract and return the metrics from a list of losses
    """
    full_loss_list = []
    for losses_list, *_ in rep_full_list:
        final_loss = [l[-1] for l in losses_list]
        full_loss_list.append(final_loss)

    full_loss_list = np.array(full_loss_list)

    return_list = [full_loss_list]

    for idx in range(1, len(rep_full_list[0])):
        return_list.append(np.array(np.array([err[idx] for err in rep_full_list])))

    return return_list


# Polytope functions
def get_polytopes(model, train_x, penultimate=False):
    """
    Returns the polytopes.
    Points that has same activations values after fed to the model
     belong to the same polytope.
    """
    polytope_memberships = []
    last_activations = train_x.cpu().numpy()
    penultimate_act = None
    layers = [module for module in model.modules() if type(module) == torch.nn.Linear]

    for layer_id, layer in enumerate(layers):
        weights, bias = (
            layer.weight.data.detach().cpu().numpy(),
            layer.bias.data.detach().cpu().numpy(),
        )
        preactivation = np.matmul(last_activations, weights.T) + bias
        if layer_id == len(layers) - 1:
            preactivation = 1 / (1 + np.exp(-1 / (1 + np.exp(-preactivation))))
            binary_preactivation = (preactivation > 0.5).astype("int")
        else:
            binary_preactivation = (preactivation > 0).astype("int")
        polytope_memberships.append(binary_preactivation)
        last_activations = preactivation * binary_preactivation

        if penultimate and layer_id == len(layers) - 1:
            penultimate_act = last_activations

    polytope_memberships = [
        np.tensordot(
            np.concatenate(polytope_memberships, axis=1),
            2
            ** np.arange(0, np.shape(np.concatenate(polytope_memberships, axis=1))[1]),
            axes=1,
        )
    ]

    if penultimate:
        return polytope_memberships, penultimate_act
    return polytope_memberships, last_activations


def binary_pattern_mat(model, train_x):

    last_activations = train_x.cpu().numpy()
    layers = [module for module in model.modules() if type(module) == torch.nn.Linear]
    for layer_id, layer in enumerate(layers):
        weights, bias = (
            layer.weight.data.detach().cpu().numpy(),
            layer.bias.data.detach().cpu().numpy(),
        )
        preactivation = np.matmul(last_activations, weights.T) + bias
        binary_preactivation = (preactivation > 0).astype("int")
        if layer_id == len(layers) - 2:
            break
        last_activations = preactivation * binary_preactivation

    binary_str = []
    for idx, pattern in enumerate(binary_preactivation):
        binary_str.append("".join(str(x) for x in pattern))

    return np.array(binary_str)


"""
  Example to run the `Increasing Depth` vs `Increasing Width` experiments
  and plot the figure.
"""
## Example
# result_d = run_deep_net_experiment(increase_depth=True, n_iterations=20, n_reps=1)
# result_w = run_deep_net_experiment(increase_width=True, n_iterations=70, n_reps=1)
# results = [result_w, result_d]
# titles = [
#             "DeepNet: Increasing Width",
#             "DeepNet: Increasing Depth",
#         ]
# plot_results(results, titles, save=True)
