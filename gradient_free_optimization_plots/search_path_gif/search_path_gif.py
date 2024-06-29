# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

import os
import glob

from .plot_search_paths import plot_search_paths


dir_ = os.path.dirname(os.path.abspath(__file__))


def search_path_gif(
    optimizer,
    n_iter,
    objective_function,
    search_space,
    path=None,
    name=None,
    constraints=None,
    initialize=None,
    opt_para=None,
    random_state=0,
    title=True,
):
    if path is None:
        path = "./gifs"
    if name is None:
        name = str(optimizer._name_) + ".gif"
    if opt_para is None:
        opt_para = {}
    if constraints is None:
        constraints = []
    if initialize is None:
        initialize = {"random": 4}

    path = os.path.join(os.getcwd(), path)

    print("\n\n name", name)
    plots_dir = path + "/_plots/"
    print(" plots_dir", plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    plot_search_paths(
        path=path,
        optimizer=optimizer,
        opt_para=opt_para,
        n_iter_max=n_iter,
        objective_function=objective_function,
        search_space=search_space,
        constraints=constraints,
        initialize=initialize,
        random_state=random_state,
        title=title,
    )

    ### ffmpeg
    framerate = str(n_iter / 10)
    # framerate = str(10)
    _framerate = " -framerate " + framerate + " "

    _input = " -i " + path + "/_plots/" + str(optimizer._name_) + "_" + "%03d.jpg "
    _scale = " -vf scale=1200:-1:flags=lanczos "
    _output = os.path.join(path, name)

    ffmpeg_command = (
        "ffmpeg -hide_banner -loglevel error -y"
        + _framerate
        + _input
        + _scale
        + _output
    )
    print("\n -----> ffmpeg_command \n", ffmpeg_command, "\n")
    print("create " + name)

    os.system(ffmpeg_command)

    ### remove _plots
    rm_files = glob.glob(path + "/_plots/*.jpg")
    for f in rm_files:
        os.remove(f)
    os.rmdir(plots_dir)
