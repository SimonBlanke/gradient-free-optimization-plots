# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import matplotlib.pyplot as plt

from ._objective import draw_objective, evaluate_objective_grid


def plot_objective_function_2d(
    objective_function, search_space, figsize=(6, 6), alpha=1
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x_all, y_all, zi = evaluate_objective_grid(objective_function, search_space)
    draw_objective(ax, x_all, y_all, zi, alpha=alpha)

    fig.tight_layout()
    return fig, ax
