# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import gc
import os

import numpy as np
from gradient_free_optimizers.optimizers.core_optimizer.converter import Converter
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

from .._objective import (
    _objective_function_np,
    draw_objective,
    evaluate_objective_grid,
)

plt.rcParams["figure.facecolor"] = "w"
mpl.use("agg")


def _draw_objective_background(ax, objective_function, search_space):
    x_all, y_all, zi = evaluate_objective_grid(objective_function, search_space)
    return draw_objective(ax, x_all, y_all, zi, alpha=0.15)


def _draw_search_paths(ax, opt, conv, n_iter):
    """Draw the visited positions of every sub-optimizer up to n_iter.

    n_iter is split across the sub-optimizers (one per particle/individual for
    population methods, a single one otherwise), so each contributes its share
    of points. Returns the last scatter, which carries the score color scale
    for the colorbar, or None if no point was drawn.
    """
    n_optimizers = len(opt.optimizers)
    scatter = None

    for n, opt_ in enumerate(opt.optimizers):
        n_iter_tmp = int(n_iter / n_optimizers)
        n_iter_mod = n_iter % n_optimizers

        if n_iter_mod > n:
            n_iter_tmp += 1
        if n_iter_tmp == 0:
            continue

        pos_list = np.array(opt_._pos_new_list)
        score_list = np.array(opt_._score_new_list)

        if len(pos_list) == 0:
            continue

        values_list = np.array(conv.positions2values(pos_list))

        ax.plot(
            values_list[:n_iter_tmp, 0],
            values_list[:n_iter_tmp, 1],
            linestyle="--",
            marker=",",
            color="black",
            alpha=0.33,
            label=n,
            linewidth=0.5,
        )
        scatter = ax.scatter(
            values_list[:n_iter_tmp, 0],
            values_list[:n_iter_tmp, 1],
            c=score_list[:n_iter_tmp],
            marker="H",
            s=15,
            vmin=np.amin(score_list[:n_iter_tmp]),
            vmax=np.amax(score_list[:n_iter_tmp]),
            label=n,
            edgecolors="black",
            linewidth=0.3,
            cmap="jet_r",
        )

    return scatter


def _set_title(ax, title, opt, opt_para, n_iter, show_opt_para):
    nth_iteration = "\n\nnth Iteration: " + str(n_iter)
    opt_para_name = ""
    opt_para_value = "\n\n"

    if show_opt_para:
        opt_para_name += "\n Parameter:"
        for para_name, para_value in opt_para.items():
            opt_para_name += "\n " + "     " + para_name + ": "
            opt_para_value += "\n " + str(para_value) + "                "

    if title is True:
        title_name = opt.name + "\n" + opt_para_name
        ax.set_title(title_name, loc="left", fontsize=18)
        ax.set_title(opt_para_value, loc="center", fontsize=15)
    elif isinstance(title, str):
        ax.set_title(title, loc="left", fontsize=18)

    ax.set_title(nth_iteration, loc="right", fontsize=10)


def _save_frame(fig, path, name_stem, n_iter):
    fig.savefig(
        os.path.join(
            path,
            "_plots",
            name_stem + "_" + "{0:0=3d}".format(n_iter) + ".jpg",
        ),
        dpi=150,
        pad_inches=0,
        bbox_inches="tight",
    )


def plot_search_path(
    title,
    opt,
    opt_para,
    objective_function,
    search_space,
    n_iter,
    conv,
    path,
    show_opt_para,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    image = _draw_objective_background(ax, objective_function, search_space)
    scatter = _draw_search_paths(ax, opt, conv, n_iter)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    _set_title(ax, title, opt, opt_para, n_iter, show_opt_para)

    mappable = scatter if scatter is not None else image
    clb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    clb.set_label("score", labelpad=-15, y=1.05, rotation=0)

    if show_opt_para:
        fig.subplots_adjust(top=0.75)

    fig.tight_layout()

    _save_frame(fig, path, opt._name_, n_iter)

    plt.close(fig)
    gc.collect()


def plot_search_paths(
    path,
    optimizer,
    opt_para,
    n_iter_max,
    objective_function,
    search_space,
    constraints,
    initialize,
    random_state,
    title,
):
    show_opt_para = bool(opt_para)

    opt = optimizer(
        search_space,
        initialize=initialize,
        constraints=constraints,
        random_state=random_state,
        **opt_para,
    )
    opt.search(
        objective_function,
        n_iter=n_iter_max,
        verbosity=False,
    )

    conv = Converter(search_space)
    for n_iter in tqdm(range(1, n_iter_max + 1)):
        plot_search_path(
            title,
            opt,
            opt_para,
            objective_function,
            search_space,
            n_iter,
            conv,
            path,
            show_opt_para,
        )
