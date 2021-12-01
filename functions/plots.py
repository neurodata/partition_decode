import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch

import copy

import seaborn as sns
from sklearn.preprocessing import minmax_scale
from matplotlib import cm

import os


def plot_decision_boundaries(
    model, num_node, num_poly, err, method="contour", depth=True
):
    """
    Plot the decision boundaries of the model
    """
    # create grid to evaluate model
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    XX, YY = np.meshgrid(
        np.arange(x_min, x_max, (x_max - x_min) / 50),
        np.arange(y_min, y_max, (y_max - y_min) / 50),
    )

    XY = np.vstack([XX.ravel(), YY.ravel()]).T

    poly_m, activations = get_polytopes(model, torch.FloatTensor(XY))

    with torch.no_grad():
        pred = model(torch.FloatTensor(XY).cuda())
        pred = torch.sigmoid(pred).detach().cpu().numpy()

    gini_list = gini_impurity_list(poly_m[0], np.round(pred))

    Z = poly_m[0].reshape(XX.shape)
    bins = np.arange(0, len(poly_m[0]))
    act_bin = np.digitize(poly_m[0], bins)

    if method == "all":
        fig, ax = plt.subplots(1, 3, figsize=(21, 5))
        for a in ax:
            a.axes.xaxis.set_visible(False)
            a.axes.yaxis.set_visible(False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    if method == "surface" or method == "all":
        m = poly_m[0]
        m = minmax_scale(m, feature_range=(0, 1), axis=0, copy=True)
        my_col = cm.tab20b(m.reshape(XX.shape))

        if method == "surface":
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(132, projection="3d")

        ax.view_init(elev=45.0, azim=15)
        ax.plot_surface(
            X=XX,
            Y=YY,
            Z=pred.reshape(XX.shape),
            facecolors=my_col,
            linewidth=0,
            antialiased=False,
            rstride=1,
            cstride=1,
        )
        # Get rid of the panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title("Generalization error: %.4f" % err)

    if method == "colormesh" or method == "all":
        if method == "all":
            ax = fig.add_subplot(131)
        plt.pcolormesh(XX, YY, Z, cmap="PRGn")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            "Nodes: " + str(num_node) + "; # of activated regions: " + str(num_poly)
        )

    # if method == 'contour' or method=='all':
    #     if method == 'all':
    #         ax = fig.add_subplot(142)

    #     plt.contourf(XX, YY, Z, cmap="tab20b",  vmin = np.min(Z), vmax = np.max(Z))
    #     ax.set_title("Nodes: " + str(num_node) + "; # of activated regions: " + str(num_poly))
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    if method == "gini" or method == "all":
        if method == "all":
            ax = fig.add_subplot(133)
        # gini_Z = minmax_scale(gini_list, feature_range=(0, 1), axis=0, copy=True)
        gini_Z = gini_list.reshape(XX.shape)
        plt.pcolormesh(XX, YY, gini_Z, cmap="Reds", vmin=0, vmax=0.5)
        cbar = plt.colorbar(ticks=np.linspace(0, 0.5, 2))
        cbar.ax.set_title("gini index", fontsize=12, pad=12)
        cbar.set_ticklabels(["0", "0.5"])
        ax.set_title("Mean: %.4f" % np.mean(gini_list))
        ax.set_xticks([])
        ax.set_yticks([])

    exp = "depth" if depth else "width"
    os.makedirs("../polytopes/", exist_ok=True)
    plt.savefig("../polytopes/xor_%s_%s_%04d.png" % (exp, method, num_node))
    # plt.show()


# Plot the result
def plot_results(results):
    """
    Generate the DeepNet: Increasing Depth vs Increasing Width figure.
    results should consist the `Increasing Width` and `Increasing Depth` results, respectively.
    """

    sns.set()
    fontsize = 20
    ticksize = 20

    bayes_err = 0.25

    # Figure params
    fontsize = 22
    ticksize = 20
    linewidth = 2
    fig, axes = plt.subplots(
        figsize=(14, 20), nrows=4, ncols=2, sharex="col", sharey="row"
    )
    # plt.figure(
    plt.tick_params(labelsize=ticksize)
    plt.tight_layout()

    titles = [
        "DeepNet: Increasing Width",
        "DeepNet: Increasing Depth",
        "DeepNet: Increasing Width (5 layers)",
    ]

    ## Average Stability, Bias and Variance
    for i in range(len(results)):
        result = results[i]

        ## You can choose the panels to display
        # metric_list = [(result.train_err_list, result.test_err_list), (result.train_loss_list, result.test_loss_list), result.penultimate_vars_reps, result.poly_list, result.briers_list, (result.gini_train, result.gini_test), result.avg_stab, result.bias, result.var]
        # metric_ylab = ["Generalization Error", "Cross-Entropy Loss", "Variance of last activation", "Activated regions", "Hellinger distance", "Gini impurity", "Average stability", "Average Bias", "Average Variance"]
        metric_list = [
            (result.train_err_list, result.test_err_list),
            (result.train_loss_list, result.test_loss_list),
            (result.gini_train, result.gini_test),
            result.poly_list,
        ]
        metric_ylab = [
            "Generalization Error",
            "Cross-Entropy Loss",
            "Gini impurity",
            "Activated regions",
            "Hellinger distance",
        ]

        for j, metric in enumerate(metric_list):
            ax = axes[j, i]
            if isinstance(metric, tuple):
                ax.plot(
                    result.num_pars,
                    np.median(metric[0], 0).clip(min=0),
                    label="Train",
                    linewidth=2,
                )
                ax.fill_between(
                    result.num_pars,
                    np.percentile(metric[0], 25, axis=0).clip(min=0),
                    np.percentile(metric[0], 75, axis=0),
                    alpha=0.2,
                )
                ax.plot(
                    result.num_pars,
                    np.median(metric[1], 0),
                    label="Test",
                    color="red",
                    linewidth=2,
                )
                ax.fill_between(
                    result.num_pars,
                    np.percentile(metric[1], 25, axis=0).clip(min=0),
                    np.percentile(metric[1], 75, axis=0),
                    alpha=0.2,
                )
            else:
                ax.plot(result.num_pars, np.median(metric, 0).clip(min=0), linewidth=2)
                ax.fill_between(
                    result.num_pars,
                    np.percentile(metric, 25, axis=0).clip(min=0),
                    np.percentile(metric, 75, axis=0),
                    alpha=0.2,
                )

            ax.axvline(x=1000, color="gray", alpha=0.6)
            if j == 0:
                ax.set_title(titles[i], fontsize=fontsize + 2)
                # ax.axhline(y=bayes_err, color='gray', linestyle='--')

            if i == 0:
                ax.set_ylabel(metric_ylab[j], fontsize=fontsize)

            ax.set_xscale("log")
            #     ax = plt.gca()
            ax.locator_params(nbins=6, axis="y")
            # ax.locator_params(nbins=6, axis='x')
            ax.tick_params(axis="both", which="major", labelsize=ticksize)

    lines, labels = ax.get_legend_handles_labels()
    plt.legend(
        lines,
        labels,
        loc="best",
        bbox_to_anchor=(0.0, -0.009, 1, 1),
        bbox_transform=plt.gcf().transFigure,
        fontsize=fontsize - 5,
        frameon=False,
    )

    # plt.text(2.8, -0.0490, 'Total parameters', ha='center', fontsize=fontsize)
    sns.despine()
    os.makedirs("../results", exist_ok=True)
    plt.savefig("../results/DeepNet.pdf", bbox_inches="tight")
