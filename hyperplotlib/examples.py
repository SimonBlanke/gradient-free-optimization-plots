# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperactive import Hyperactive
import scipy.interpolate


def search_path_2d(
    search_config, X, y, n_iter, optimizer, random_state=False, ax=None, figsize=None
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    opt = Hyperactive(X, y, memory="long", random_state=5)
    opt.search(search_config, n_iter=n_iter, optimizer=optimizer)

    model = list(search_config.keys())[0]
    dim_names = list(search_config[model].keys())

    pos_list = opt.pos_list[model]
    score_list = opt.score_list[model]

    pos = pos_list[0]
    score = score_list[0]

    x_1d = list(np.arange(0, 100, 0.1))
    y_1d = list(np.arange(0, 100, 0.1))
    x_2d, y_2d = np.meshgrid(x_1d, y_1d)

    para_temp = {}
    para_temp[dim_names[0]] = x_2d
    para_temp[dim_names[1]] = y_2d

    z_2d = model(para_temp, X, y)

    df = pd.DataFrame({"X": pos[:, 0], "Y": pos[:, 1], "score": score})

    plt.set_cmap("Greys")
    plt.pcolormesh(x_2d, y_2d, z_2d)
    plt.colorbar()
    plt.plot("X", "Y", data=df, linestyle="--", marker=",", color="black", alpha=0.99)

    return ax
