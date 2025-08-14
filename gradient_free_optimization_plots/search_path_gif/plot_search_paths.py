# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Dict, Any, Callable, List, Union, Optional
import gc
import numpy as np
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from gradient_free_optimizers.optimizers.core_optimizer.converter import Converter

plt.rcParams["figure.facecolor"] = "w"
mpl.use("agg")


class PlotConfig:
    FIGURE_SIZE = (7, 7)
    DPI = 350
    COLORMAP = "jet_r"
    ALPHA = 0.15
    LINE_WIDTH = 0.5
    MARKER_SIZE = 15
    EDGE_LINE_WIDTH = 0.3
    TITLE_FONT_SIZE = 18
    PARAM_FONT_SIZE = 15
    ITER_FONT_SIZE = 10


def _objective_function_np(objective_function: Callable, search_space: Dict[str, Any]) -> Callable:
    """Convert objective function to work with numpy arrays."""
    def wrapper(args):
        params = {}
        for i, para_name in enumerate(search_space):
            params[para_name] = args[i]
        return objective_function(params)
    return wrapper


def _cleanup_matplotlib() -> None:
    """Clean up matplotlib resources."""
    plt.ioff()
    plt.cla()
    plt.clf()
    plt.close("all")
    gc.collect()


def _create_contour_plot(objective_function: Callable, search_space: Dict[str, Any]) -> None:
    """Create the contour plot background."""
    objective_func_np = _objective_function_np(objective_function, search_space)
    
    dim0_key, dim1_key = list(search_space.keys())
    x_all, y_all = search_space[dim0_key], search_space[dim1_key]
    xi, yi = np.meshgrid(x_all, y_all)
    zi = objective_func_np((xi, yi))
    
    plt.imshow(
        zi,
        alpha=PlotConfig.ALPHA,
        extent=[x_all.min(), x_all.max(), y_all.min(), y_all.max()],
    )


def _plot_optimizer_paths(opt: Any, n_iter: int, conv: Converter) -> None:
    """Plot the search paths for all optimizers."""
    for n, opt_ in enumerate(opt.optimizers):
        n_optimizers = len(opt.optimizers)
        n_iter_tmp = int(n_iter / n_optimizers)
        n_iter_mod = n_iter % n_optimizers

        if n_iter_mod > n:
            n_iter_tmp += 1
        if n_iter_tmp == 0:
            continue

        pos_list = np.array(opt_.pos_new_list)
        score_list = np.array(opt_.score_new_list)

        if len(pos_list) == 0:
            continue

        values_list = conv.positions2values(pos_list)
        values_list = np.array(values_list)

        plt.plot(
            values_list[:n_iter_tmp, 0],
            values_list[:n_iter_tmp, 1],
            linestyle="--",
            marker=",",
            color="black",
            alpha=0.33,
            label=str(n),
            linewidth=PlotConfig.LINE_WIDTH,
        )

        try:
            plt.scatter(
                values_list[:n_iter_tmp, 0],
                values_list[:n_iter_tmp, 1],
                c=score_list[:n_iter_tmp],
                marker="H",
                s=PlotConfig.MARKER_SIZE,
                vmin=np.amin(score_list[:n_iter_tmp]),
                vmax=np.amax(score_list[:n_iter_tmp]),
                label=str(n),
                edgecolors="black",
                linewidth=PlotConfig.EDGE_LINE_WIDTH,
            )
        except Exception as e:
            print(f"Error plotting scatter: values_list shape: {values_list.shape}, score_list shape: {score_list.shape}")
            raise ValueError(f"Failed to create scatter plot: {e}")


def _set_titles_and_labels(title: Union[str, bool], opt: Any, opt_para: Dict[str, Any], 
                          n_iter: int, show_opt_para: bool) -> None:
    """Set plot titles and labels."""
    plt.xlabel("x")
    plt.ylabel("y")

    nth_iteration = f"\n\nnth Iteration: {n_iter}"
    opt_para_name = ""
    opt_para_value = "\n\n"

    if show_opt_para:
        opt_para_name += "\n Parameter:"
        for para_name, para_value in opt_para.items():
            opt_para_name += f"\n      {para_name}: "
            opt_para_value += f"\n {para_value}                "

    if title is True:
        title_name = opt.name + "\n" + opt_para_name
        plt.title(title_name, loc="left", fontsize=PlotConfig.TITLE_FONT_SIZE)
        plt.title(opt_para_value, loc="center", fontsize=PlotConfig.PARAM_FONT_SIZE)
    elif isinstance(title, str):
        plt.title(title, loc="left", fontsize=PlotConfig.TITLE_FONT_SIZE)

    plt.title(nth_iteration, loc="right", fontsize=PlotConfig.ITER_FONT_SIZE)

    clb = plt.colorbar(fraction=0.046, pad=0.04)
    clb.set_label("score", labelpad=-15, y=1.05, rotation=0)

    if show_opt_para:
        plt.subplots_adjust(top=0.75)

    plt.tight_layout()


def plot_search_path(
    title: Union[str, bool],
    opt: Any,
    opt_para: Dict[str, Any],
    objective_function: Callable,
    search_space: Dict[str, Any],
    n_iter: int,
    conv: Converter,
    path: str,
    show_opt_para: bool,
) -> None:
    """Plot a single frame of the search path animation."""

    plt.figure(figsize=PlotConfig.FIGURE_SIZE)
    plt.set_cmap(PlotConfig.COLORMAP)

    _create_contour_plot(objective_function, search_space)
    _plot_optimizer_paths(opt, n_iter, conv)
    _set_titles_and_labels(title, opt, opt_para, n_iter, show_opt_para)

    filename = f"{opt._name_}_{n_iter:03d}.jpg"
    filepath = f"{path}/_plots/{filename}"
    
    plt.savefig(
        filepath,
        dpi=PlotConfig.DPI,
        pad_inches=0,
        bbox_inches="tight",
    )

    _cleanup_matplotlib()


def plot_search_paths(
    path: str,
    optimizer: Any,
    opt_para: Dict[str, Any],
    n_iter_max: int,
    objective_function: Callable,
    search_space: Dict[str, Any],
    constraints: Optional[List] = None,
    initialize: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    title: Union[str, bool] = True,
) -> None:
    """Generate all frames for the search path animation."""
    show_opt_para = bool(opt_para)

    opt = optimizer(
        search_space,
        initialize=initialize,
        constraints=constraints,
        random_state=random_state,
        **opt_para
    )
    opt.search(
        objective_function,
        n_iter=n_iter_max,
        verbosity=False,
    )

    conv = Converter(search_space)
    for n_iter in tqdm(range(1, n_iter_max + 1), desc="Generating plots"):
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
