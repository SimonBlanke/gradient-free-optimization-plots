# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import glob
import os
import subprocess

from .plot_search_paths import plot_search_paths


class SearchPathGif:
    def __init__(self, path=None) -> None:
        if path is None:
            path = "./gifs"
        path = os.path.join(os.getcwd(), path)

        self.path = path

    def add_optimizer(
        self, optimizer, n_iter, opt_para=None, initialize=None, random_state=None
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

    def add_test_function(self, objective_function, search_space, constraints=None):
        if constraints is None:
            constraints = []

        self.objective_function = objective_function
        self.search_space = search_space
        self.constraints = constraints

    def add_plot_layout(self, name=None, title=None):
        if name is None:
            name = str(self.optimizer._name_) + ".gif"

        self.name = name
        self.title = title

    def create(self):
        plots_dir = os.path.join(self.path, "_plots")
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

        framerate = str(self.n_iter / 10)
        input_pattern = os.path.join(
            plots_dir, str(self.optimizer._name_) + "_%03d.jpg"
        )
        output_path = os.path.join(self.path, self.name)

        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-framerate", framerate,
                "-i", input_pattern,
                "-vf", "scale=1200:-1:flags=lanczos",
                output_path,
            ],
            check=True,
        )

        rm_files = glob.glob(os.path.join(plots_dir, "*.jpg"))
        for f in rm_files:
            os.remove(f)
        os.rmdir(plots_dir)
