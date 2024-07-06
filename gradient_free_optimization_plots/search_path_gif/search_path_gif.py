# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .plot_search_paths import plot_search_paths
import glob
import os
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


dir_ = os.path.dirname(os.path.abspath(__file__))


class SearchPathGif:
    def __init__(self, path) -> None:
        if path is None:
            path = "./gifs"
        path = os.path.join(os.getcwd(), path)

        self.path = path

    def add_optimizer(
        self, optimizer, opt_para, initialize, n_iter, random_state
    ):
        if opt_para is None:
            opt_para = {}
        if initialize is None:
            initialize = {"random": 4}

        self.optimizer = optimizer
        self.opt_para = opt_para
        self.initialize = initialize
        self.n_iter = n_iter
        self.random_state = random_state

    def add_test_function(self, objective_function, search_space, constraints):
        if constraints is None:
            constraints = []

        self.objective_function = objective_function
        self.search_space = search_space
        self.constraints = constraints

    def add_plot_layout(self, name, title):
        if name is None:
            name = str(self.optimizer._name_) + ".gif"

        self.name = name
        self.title = title

    def create(self):
        print("\n\n name", self.name)
        plots_dir = self.path + "/_plots/"
        print(" plots_dir", plots_dir)
        os.makedirs(plots_dir, exist_ok=True)

        plot_search_paths(
            path=self.path,
            optimizer=self.optimizer,
            opt_para=self.opt_para,
            n_iter_max=self.n_iter,
            objective_function=self.objective_function,
            search_space=self.search_space,
            constraints=self.constraints,
            initialize=self.initialize,
            random_state=self.random_state,
            title=self.title,
        )

        # ffmpeg
        framerate = str(self.n_iter / 10)
        # framerate = str(10)
        _framerate = " -framerate " + framerate + " "

        _input = (
            " -i "
            + self.path
            + "/_plots/"
            + str(self.optimizer._name_)
            + "_"
            + "%03d.jpg "
        )
        _scale = " -vf scale=1200:-1:flags=lanczos "
        _output = os.path.join(self.path, self.name)

        ffmpeg_command = (
            "ffmpeg -hide_banner -loglevel error -y"
            + _framerate
            + _input
            + _scale
            + _output
        )
        print("\n -----> ffmpeg_command \n", ffmpeg_command, "\n")
        print("create " + self.name)

        os.system(ffmpeg_command)

        # remove _plots
        rm_files = glob.glob(self.path + "/_plots/*.jpg")
        for f in rm_files:
            os.remove(f)
        os.rmdir(plots_dir)
