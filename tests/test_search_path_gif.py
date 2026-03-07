import os
import tempfile
from unittest.mock import patch, MagicMock

import matplotlib
import numpy as np
import pytest

matplotlib.use("agg")

from gradient_free_optimizers import HillClimbingOptimizer

from gradient_free_optimization_plots import SearchPathGif


class TestSearchPathGifInit:
    def test_default_path(self):
        spg = SearchPathGif()
        assert spg.path.endswith("gifs")

    def test_custom_path(self):
        spg = SearchPathGif(path="my_output")
        assert spg.path.endswith("my_output")

    def test_none_path_uses_default(self):
        spg = SearchPathGif(path=None)
        assert spg.path.endswith("gifs")


class TestSearchPathGifAddOptimizer:
    def test_stores_optimizer(self):
        spg = SearchPathGif()
        spg.add_optimizer(
            optimizer=HillClimbingOptimizer,
            n_iter=10,
        )
        assert spg.optimizer is HillClimbingOptimizer
        assert spg.n_iter == 10
        assert spg.opt_para == {}
        assert spg.initialize == {"random": 4}
        assert spg.random_state is None

    def test_custom_parameters(self):
        spg = SearchPathGif()
        spg.add_optimizer(
            optimizer=HillClimbingOptimizer,
            n_iter=50,
            opt_para={"epsilon": 0.1},
            initialize={"random": 2},
            random_state=42,
        )
        assert spg.n_iter == 50
        assert spg.opt_para == {"epsilon": 0.1}
        assert spg.initialize == {"random": 2}
        assert spg.random_state == 42


class TestSearchPathGifAddTestFunction:
    def test_stores_function_and_space(self, sphere_function, search_space):
        spg = SearchPathGif()
        spg.add_test_function(
            objective_function=sphere_function,
            search_space=search_space,
        )
        assert spg.objective_function is sphere_function
        assert spg.search_space is search_space
        assert spg.constraints == []

    def test_custom_constraints(self, sphere_function, search_space):
        constraint = lambda para: para["x0"] > 0
        spg = SearchPathGif()
        spg.add_test_function(
            objective_function=sphere_function,
            search_space=search_space,
            constraints=[constraint],
        )
        assert len(spg.constraints) == 1


class TestSearchPathGifAddPlotLayout:
    def test_custom_name(self):
        spg = SearchPathGif()
        spg.add_plot_layout(name="test.gif", title="My Title")
        assert spg.name == "test.gif"
        assert spg.title == "My Title"

    def test_default_name_from_optimizer(self):
        spg = SearchPathGif()
        spg.add_optimizer(optimizer=HillClimbingOptimizer, n_iter=10)
        spg.add_plot_layout(name=None, title=None)
        assert spg.name.endswith(".gif")


class TestSearchPathGifCreate:
    @patch("gradient_free_optimization_plots.search_path_gif.search_path_gif.subprocess.run")
    @patch("gradient_free_optimization_plots.search_path_gif.search_path_gif.plot_search_paths")
    def test_create_calls_plot_and_ffmpeg(
        self, mock_plot, mock_subprocess, sphere_function, search_space
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spg = SearchPathGif(path=tmpdir)
            spg.add_optimizer(
                optimizer=HillClimbingOptimizer,
                n_iter=10,
            )
            spg.add_test_function(
                objective_function=sphere_function,
                search_space=search_space,
            )
            spg.add_plot_layout(name="test.gif", title=True)
            spg.create()

            mock_plot.assert_called_once()
            mock_subprocess.assert_called_once()

            call_args = mock_subprocess.call_args
            cmd = call_args[0][0]
            assert cmd[0] == "ffmpeg"
            assert call_args[1]["check"] is True

    @patch("gradient_free_optimization_plots.search_path_gif.search_path_gif.subprocess.run")
    @patch("gradient_free_optimization_plots.search_path_gif.search_path_gif.plot_search_paths")
    def test_create_cleans_up_plots_dir(
        self, mock_plot, mock_subprocess, sphere_function, search_space
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            spg = SearchPathGif(path=tmpdir)
            spg.add_optimizer(
                optimizer=HillClimbingOptimizer,
                n_iter=10,
            )
            spg.add_test_function(
                objective_function=sphere_function,
                search_space=search_space,
            )
            spg.add_plot_layout(name="test.gif", title=True)
            spg.create()

            plots_dir = os.path.join(tmpdir, "_plots")
            assert not os.path.exists(plots_dir)
