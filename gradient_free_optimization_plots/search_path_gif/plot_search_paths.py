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

plt.rcParams["figure.facecolor"] = "w"
mpl.use("agg")


def _objective_function_np(objective_function, search_space, args):
    params = {}
    for i, para_name in enumerate(search_space):
        params[para_name] = args[i]
    return objective_function(params)


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
    plt.figure(figsize=(7, 7))
    plt.set_cmap("jet_r")

    x_all, y_all = search_space["x0"], search_space["x1"]
    xi, yi = np.meshgrid(x_all, y_all)
    zi = _objective_function_np(objective_function, search_space, (xi, yi))
    zi = np.rot90(zi, k=1)

    plt.imshow(
        zi,
        alpha=0.15,
        extent=[x_all.min(), x_all.max(), y_all.min(), y_all.max()],
    )

    for n, opt_ in enumerate(opt.optimizers):
        n_optimizers = len(opt.optimizers)
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

        plt.plot(
            values_list[:n_iter_tmp, 0],
            values_list[:n_iter_tmp, 1],
            linestyle="--",
            marker=",",
            color="black",
            alpha=0.33,
            label=n,
            linewidth=0.5,
        )
        plt.scatter(
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
        )

    plt.xlabel("x")
    plt.ylabel("y")

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
        plt.title(title_name, loc="left", fontsize=18)
        plt.title(opt_para_value, loc="center", fontsize=15)
    elif isinstance(title, str):
        plt.title(title, loc="left", fontsize=18)

    plt.title(nth_iteration, loc="right", fontsize=10)

    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.set_label("score", labelpad=-15, y=1.05, rotation=0)

    if show_opt_para:
        plt.subplots_adjust(top=0.75)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            path,
            "_plots",
            opt._name_ + "_" + "{0:0=3d}".format(n_iter) + ".jpg",
        ),
        dpi=150,
        pad_inches=0,
        bbox_inches="tight",
    )

    plt.ioff()
    plt.cla()
    plt.clf()
    plt.close("all")
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
