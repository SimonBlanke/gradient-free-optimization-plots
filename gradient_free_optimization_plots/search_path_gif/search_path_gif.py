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

        self.width = 1200
        self.duration = 10.0
        self.dpi = 150
        self.colors = None

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

    def configure_output(self, width=None, duration=None, dpi=None, colors=None):
        """Control size, speed and file size of the produced GIF.

        width is the final pixel width; the height keeps the aspect ratio.
        duration is the total play time in seconds, and the frame rate follows
        from it as n_iter / duration, so a run keeps the same viewing time no
        matter how many iterations it shows. dpi is the rendering resolution of
        each frame; raise it together with width for a large, sharp GIF. colors
        is a palette size between 2 and 256; when set the GIF is re-encoded with
        an optimized palette, which gives a smaller file with better colors,
        while None writes the GIF in one pass as before. Only the arguments you
        pass are changed.
        """
        if width is not None:
            if int(width) <= 0:
                raise ValueError(f"width must be positive, got {width}")
            self.width = int(width)
        if duration is not None:
            if float(duration) <= 0:
                raise ValueError(f"duration must be positive, got {duration}")
            self.duration = float(duration)
        if dpi is not None:
            if int(dpi) <= 0:
                raise ValueError(f"dpi must be positive, got {dpi}")
            self.dpi = int(dpi)
        if colors is not None:
            if not 2 <= int(colors) <= 256:
                raise ValueError(f"colors must be in 2..256, got {colors}")
            self.colors = int(colors)

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
            dpi=self.dpi,
        )

        # the frame rate follows from the wanted total play time, so the GIF
        # runs for `duration` seconds regardless of how many iterations it shows
        framerate = str(self.n_iter / self.duration)
        input_pattern = os.path.join(
            plots_dir, str(self.optimizer._name_) + "_%03d.jpg"
        )
        output_path = os.path.join(self.path, self.name)

        self._assemble_gif(plots_dir, framerate, input_pattern, output_path)

        for f in glob.glob(os.path.join(plots_dir, "*.jpg")):
            os.remove(f)
        palette_path = os.path.join(plots_dir, "_palette.png")
        if os.path.exists(palette_path):
            os.remove(palette_path)
        os.rmdir(plots_dir)

    def _assemble_gif(self, plots_dir, framerate, input_pattern, output_path):
        scale = f"scale={self.width}:-1:flags=lanczos"

        if self.colors is None:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    framerate,
                    "-i",
                    input_pattern,
                    "-vf",
                    scale,
                    output_path,
                ],
                check=True,
            )
            return

        # build an optimal palette from all frames, then re-encode using it;
        # this gives a smaller file with better colors than ffmpeg's default
        palette_path = os.path.join(plots_dir, "_palette.png")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                framerate,
                "-i",
                input_pattern,
                "-vf",
                f"{scale},palettegen=max_colors={self.colors}",
                palette_path,
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                framerate,
                "-i",
                input_pattern,
                "-i",
                palette_path,
                "-lavfi",
                f"{scale}[x];[x][1:v]paletteuse",
                output_path,
            ],
            check=True,
        )
