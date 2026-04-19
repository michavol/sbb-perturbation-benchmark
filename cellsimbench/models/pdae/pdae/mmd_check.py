# https://github.com/antoninschrab/mmdagg/tree/master

import os
import torch
import numpy as np
import jax.numpy as jnp
from mmdagg.jax import mmdagg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns



def _to_jax(x):
    if not isinstance(x, jnp.ndarray):
        if isinstance(x, torch.Tensor):
            x_jax = jnp.array(x.cpu())
        else:
            x_jax = jnp.array(x)
        return x_jax
    else:
        return x


def _to_numpy(x):
    if not isinstance(x, np.ndarray):
        if isinstance(x, torch.Tensor):
            x_numpy = x.cpu().numpy()
        else:
            x_numpy = np.array(x)
        return x_numpy
    else:
        return x


def run_mmdagg_test(x, y, alpha, kernel="laplace_gaussian", number_bandwidths=10, seed=0):
    x = _to_jax(x)
    y = _to_jax(y)
    reject, details = mmdagg(x, y, alpha=alpha, kernel=kernel, number_bandwidths=number_bandwidths, seed=seed, return_dictionary=True)
    return reject, details


def get_mmdagg_test_results(x, y, alpha=0.05):
    
    assert x.shape[-1] == y.shape[-1]
    
    reject, details = run_mmdagg_test(x,y, alpha)
    
    pvals = []
    thresholds = []
    rejects = []
    for key in details.keys():
        if "Single test" in key:
            rejects.append(int(details[key]["Reject"].item()))
            pvals.append(float(details[key]["p-value"]))
            thresholds.append(float(details[key]["p-value threshold"]))
    
    pvals = jnp.array(pvals)
    rejects = jnp.array(rejects)

    return {
        "reject": reject.astype(bool).item(),
        "num_rejects": rejects.sum().item(),
        "num_tests": rejects.shape[0],
        "ratio_rejects": (rejects.sum() / rejects.shape[0]).item(),
        "p_min": jnp.min(pvals).item(),
        "p_median": jnp.median(pvals).item(),
        "p_max": jnp.max(pvals).item(),
        "thresholds": thresholds,
        "pvals": pvals,
    }


def make_mmdagg_distribution_plot(x, y, alpha, plot_path, plot_save_name, title, xlabel, ylabel):
    """
    Create a distribution plot comparing two sets of data using MMD-AGG.
    Visualizes the distributions and includes MMD test results in the plot.
    """
    x = _to_numpy(x)
    y = _to_numpy(y)

    stats_dict = get_mmdagg_test_results(x, y, alpha=alpha)

    sns.set_style("darkgrid")
    blue = sns.color_palette()[0]
    orange = sns.color_palette()[1]

    if x.shape[-1] == y.shape[-1] == 2:
        # Use gridspec for 2D KDE with marginals
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(4, 4)

        ax_main = fig.add_subplot(gs[1:4, 0:3])
        ax_xdist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
        ax_ydist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

        # 2D KDE plots
        sns.kdeplot(x=x[:,0], y=x[:,1], ax=ax_main, color=blue, label=xlabel)
        sns.kdeplot(x=y[:,0], y=y[:,1], ax=ax_main, color=orange, label=ylabel)
        # sns.kdeplot(x=x[:, 0], y=x[:, 1], ax=ax_main, fill=True, alpha=0.8, cmap="Blues", thresh=0.05, levels=10)
        # sns.kdeplot(x=y[:, 0], y=y[:, 1], ax=ax_main, fill=True, alpha=0.4, cmap="Oranges", thresh=0.05, levels=10)


        # 1D marginal KDEs
        sns.kdeplot(x[:,0], ax=ax_xdist, color=blue, fill=True)
        sns.kdeplot(y[:,0], ax=ax_xdist, color=orange, fill=True)
        ax_xdist.axis("off")

        sns.kdeplot(x[:,1], ax=ax_ydist, color=blue, fill=True, vertical=True)
        sns.kdeplot(y[:,1], ax=ax_ydist, color=orange, fill=True, vertical=True)
        ax_ydist.axis("off")

        ax = ax_main  # Use for annotations and title
    elif len(x.shape) == len(y.shape) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        sns.kdeplot(x, label=xlabel, ax=ax, color=blue)
        sns.kdeplot(y, label=ylabel, ax=ax, color=orange)
    else:
        raise NotImplementedError

    # Stats text
    textstr = '\n'.join((
        f"Reject: {stats_dict['reject']}",
        f"# Rejects: {stats_dict['num_rejects']}",
        f"# Tests: {stats_dict['num_tests']}",
        f"Ratio: {stats_dict['ratio_rejects']:.2f}",
    ))

    ax.text(
        0.95, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )

    # Custom legend
    true_patch = mpatches.Patch(color=blue, label=xlabel)
    pred_patch = mpatches.Patch(color=orange, label=ylabel)
    ax.legend(handles=[true_patch, pred_patch], loc='upper left')

    ax.set_title(title)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, f'{plot_save_name}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(plot_path, f'{plot_save_name}.png'), format='png', bbox_inches='tight')
    plt.show()

