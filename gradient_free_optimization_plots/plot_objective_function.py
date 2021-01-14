# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import matplotlib.pyplot as plt


def plot_objective_function_2d(
    objective_function, search_space, figsize=(6, 6), alpha=1
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    def objective_function_np(*args):
        para = {}
        for arg, key in zip(args, search_space.keys()):
            para[key] = arg

        return objective_function(para)

    (x_all, y_all) = search_space.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = objective_function_np(xi, yi)

    plt.set_cmap("jet_r")

    plt.imshow(
        zi,
        alpha=alpha,
        extent=[x_all.min(), x_all.max(), y_all.min(), y_all.max()],
    )

    plt.tight_layout()
    return ax
