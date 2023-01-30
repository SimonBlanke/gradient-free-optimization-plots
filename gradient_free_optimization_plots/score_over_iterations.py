import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

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
    objective_function,
    search_space,
    constraints=None,
    initialize=None,
    opt_para=None,
    random_state=0,
):
    if opt_para is None:
        opt_para = {}
    if constraints is None:
        constraints = []
    if initialize is None:
        initialize = {"random": 4}

    search_data_d = {}

    for optimizer in optimizer_l:
        opt = optimizer(search_space, random_state=random_state)
        opt.search(objective_function, n_iter)

        search_data_d[opt.name] = opt.search_data

    best_score_iter_d = {}
    for opt_name in search_data_d.keys():
        print("opt_name", opt_name)
        search_data = search_data_d[opt_name]
        best_score_iter_d[opt_name] = get_best_scores_iter(search_data)
    print("\n best_score_iter_d \n", best_score_iter_d, "\n")

    df_d = {}

    for opt_name in best_score_iter_d.keys():
        df = pd.DataFrame(best_score_iter_d[opt_name], columns=["score"])
        df["nth iteration"] = list(range(1, len(df) + 1))
        df["Optimizer"] = opt_name

        df_d[opt_name] = df

    """
    plot_data = pd.concat(list(df_d.values()))
    print("\n plot_data \n", plot_data, "\n")

    fig = px.line(plot_data, x="nth iteration", y="score", color="Optimizer")
    fig.show()
    """


def score_over_iter_plot_error_bands(
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
        print("\n\n", optimizer.name)
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

    scatter_plot_l = []

    for idx, opt_name in enumerate(plot_data_d.keys()):
        plot_data = plot_data_d[opt_name]

        color = DEFAULT_PLOTLY_COLORS[idx]

        scatter_plot = go.Scatter(
            name=opt_name,
            x=plot_data["nth iteration"].values,
            y=-plot_data["average score"].values,
            mode="lines",
            line=dict(color=color),
        )
        scatter_plot_l.append(scatter_plot)

        if error_bands:
            scatter_plot_upper = go.Scatter(
                name="Upper Bound",
                x=plot_data["nth iteration"],
                y=-(plot_data["average score"] + plot_data["standard deviation"]),
                mode="lines",
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False,
            )
            scatter_plot_l.append(scatter_plot_upper)

            scatter_plot_lower = go.Scatter(
                name="Lower Bound",
                x=plot_data["nth iteration"].values,
                y=-(
                    plot_data["average score"].values
                    - plot_data["standard deviation"].values
                ),
                marker=dict(color="#444"),
                line=dict(width=0),
                mode="lines",
                fillcolor=rgb_to_rgba(scatter_plot.line.color, 0.1),
                fill="tonexty",
                showlegend=False,
            )
            scatter_plot_l.append(scatter_plot_lower)

    fig = go.Figure(scatter_plot_l)
    fig.update_yaxes(type="log")
    fig.show()
