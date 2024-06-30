import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


DEFAULT_PLOTLY_COLORS = [
    "rgb(31, 119, 180)",
    "rgb(255, 127, 14)",
    "rgb(44, 160, 44)",
    "rgb(214, 39, 40)",
    "rgb(148, 103, 189)",
    "rgb(140, 86, 75)",
    "rgb(227, 119, 194)",
    "rgb(127, 127, 127)",
    "rgb(188, 189, 34)",
    "rgb(23, 190, 207)",
    "rgb(255, 255, 255)",
    "rgb(0, 0, 0)",
]


def rgb_to_rgba(rgb_value, alpha):
    # from https://stackoverflow.com/questions/56971587/plotly-function-to-convert-an-rgb-colour-to-rgba-python
    """
    Adds the alpha channel to an RGB Value and returns it as an RGBA Value
    :param rgb_value: Input RGB Value
    :param alpha: Alpha Value to add  in range [0,1]
    :return: RGBA Value
    """
    return f"rgba{rgb_value[3:-1]}, {alpha})"


def get_best_scores_iter(search_data):
    best_score = -np.inf

    best_scores_iter = []
    score_l = list(search_data["score"].values)

    for score_ in score_l:
        if score_ > best_score:
            best_scores_iter.append(score_)
            best_score = score_
        else:
            best_scores_iter.append(best_score)

    return pd.DataFrame(best_scores_iter, columns=["score"])


def score_over_iter_plot(
    optimizer_l,
    n_iter,
    n_runs,
    objective_function,
    search_space,
    constraints=None,
    initialize=None,
    opt_para=None,
    random_state=10,
    error_bands=True,
):
    if opt_para is None:
        opt_para = {}
    if constraints is None:
        constraints = []
    if initialize is None:
        initialize = {"vertices": 1}

    search_data_d = {}

    for optimizer in optimizer_l:
        print("\n", optimizer.name)
        search_data_d[optimizer.name] = []
        for run in range(n_runs):
            opt = optimizer(
                search_space,
                random_state=random_state + run,
                initialize=initialize,
            )
            opt.search(objective_function, n_iter, verbosity=False)

            df = opt.search_data
            # df["nth iteration"] = list(range(1, len(df) + 1))

            search_data_d[opt.name].append(get_best_scores_iter(df))

    plot_data_d = {}
    for opt_name in search_data_d.keys():
        plot_data_l = search_data_d[opt_name]

        df_concat = pd.concat(plot_data_l)
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        df_stds = by_row_index.std()

        means_l = list(np.squeeze(df_means.values))
        stds_l = list(np.squeeze(df_stds.values))

        df_dict = {
            "nth iteration": list(range(1, len(df_means) + 1)),
            "average score": means_l,
            "standard deviation": stds_l,
        }

        df = pd.DataFrame(df_dict)
        plot_data_d[opt_name] = df

    f, ax = plt.subplots(1)
    # plt.yscale("log")

    worst_value = np.inf
    for optimizer in optimizer_l:
        x = plot_data_d[optimizer.name]["nth iteration"]
        y = plot_data_d[optimizer.name]["average score"]
        e = plot_data_d[optimizer.name]["standard deviation"]

        plt.plot(x, y, label=optimizer.name)
        # plt.fill_between(x, y - e, y + e, alpha=0.1)

        y_min = np.amin(y)
        if y_min < worst_value:
            worst_value = y_min

    plt.gca().invert_yaxis()

    print("worst_value", worst_value)

    ax.set_ylim(bottom=0, top=worst_value + worst_value * 0.01)

    plt.legend(loc="upper right", bbox_to_anchor=(1.01, 1.01))
    plt.tight_layout()

    # plt.margins(0, 0)
    plt.savefig(
        "test_img",
        dpi=250,
        pad_inches=0,
        # bbox_inches="tight",
    )
