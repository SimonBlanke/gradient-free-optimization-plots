# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


def _objective_function_np(objective_function, search_space, args):
    params = {}
    for i, para_name in enumerate(search_space):
        params[para_name] = args[i]
    return objective_function(params)


def evaluate_objective_grid(objective_function, search_space):
    """Evaluate objective_function on the 2D grid spanned by the search space.

    Returns (x_all, y_all, zi) where x_all and y_all are the two dimension
    arrays and zi[i, j] = objective_function({dim0: x_all[j], dim1: y_all[i]}).
    The objective must accept numpy arrays as its parameter values, because the
    whole grid is evaluated in a single call rather than point by point.
    """
    dimensions = list(search_space)
    if len(dimensions) != 2:
        raise ValueError(
            "objective-function heatmap needs a 2D search space, got "
            f"{len(dimensions)} dimensions: {dimensions}"
        )

    x_all = search_space[dimensions[0]]
    y_all = search_space[dimensions[1]]
    xi, yi = np.meshgrid(x_all, y_all)
    zi = _objective_function_np(objective_function, search_space, (xi, yi))
    return x_all, y_all, zi


def draw_objective(ax, x_all, y_all, zi, alpha=1.0, cmap="jet_r"):
    """Draw zi as a background heatmap on ax and return the image.

    zi is indexed [row=y, col=x] as produced by meshgrid. origin="lower" maps
    it onto the axes without rotation, so the heatmap shares the Cartesian
    coordinate system of any points scattered on top. This single place defines
    the orientation; an earlier np.rot90 in the caller transposed the heatmap
    against the points (invisible only for x/y-symmetric functions).
    """
    return ax.imshow(
        zi,
        alpha=alpha,
        origin="lower",
        cmap=cmap,
        extent=[x_all.min(), x_all.max(), y_all.min(), y_all.max()],
    )
