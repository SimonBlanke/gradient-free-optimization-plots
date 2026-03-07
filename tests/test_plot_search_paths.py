import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("agg")

from gradient_free_optimizers import HillClimbingOptimizer

from gradient_free_optimization_plots.search_path_gif.plot_search_paths import (
    _objective_function_np,
    plot_search_path,
    plot_search_paths,
)
from gradient_free_optimizers.optimizers.core_optimizer.converter import Converter


class TestObjectiveFunctionNp:
    def test_scalar_inputs(self):
        def obj(para):
            return para["x0"] + para["x1"]

        search_space = {"x0": np.arange(0, 10), "x1": np.arange(0, 10)}
        result = _objective_function_np(obj, search_space, (3.0, 4.0))
        assert result == 7.0

    def test_array_inputs(self):
        def obj(para):
            return para["x0"] ** 2 + para["x1"] ** 2

        search_space = {"x0": np.arange(0, 10), "x1": np.arange(0, 10)}
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        result = _objective_function_np(obj, search_space, (x, y))
        np.testing.assert_array_equal(result, [10.0, 20.0])


class TestPlotSearchPaths:
    def test_generates_plot_files(self, sphere_function, search_space):
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "_plots")
            os.makedirs(plots_dir)

            plot_search_paths(
                path=tmpdir,
                optimizer=HillClimbingOptimizer,
                opt_para={},
                n_iter_max=3,
                objective_function=sphere_function,
                search_space=search_space,
                constraints=[],
                initialize={"random": 1},
                random_state=0,
                title=True,
            )

            jpg_files = [f for f in os.listdir(plots_dir) if f.endswith(".jpg")]
            assert len(jpg_files) == 3
            plt.close("all")

    def test_string_title(self, sphere_function, search_space):
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "_plots")
            os.makedirs(plots_dir)

            plot_search_paths(
                path=tmpdir,
                optimizer=HillClimbingOptimizer,
                opt_para={},
                n_iter_max=2,
                objective_function=sphere_function,
                search_space=search_space,
                constraints=[],
                initialize={"random": 1},
                random_state=0,
                title="Custom Title",
            )

            jpg_files = [f for f in os.listdir(plots_dir) if f.endswith(".jpg")]
            assert len(jpg_files) == 2
            plt.close("all")

    def test_with_opt_para(self, sphere_function, search_space):
        with tempfile.TemporaryDirectory() as tmpdir:
            plots_dir = os.path.join(tmpdir, "_plots")
            os.makedirs(plots_dir)

            plot_search_paths(
                path=tmpdir,
                optimizer=HillClimbingOptimizer,
                opt_para={"epsilon": 0.05},
                n_iter_max=2,
                objective_function=sphere_function,
                search_space=search_space,
                constraints=[],
                initialize={"random": 1},
                random_state=0,
                title=True,
            )

            jpg_files = [f for f in os.listdir(plots_dir) if f.endswith(".jpg")]
            assert len(jpg_files) == 2
            plt.close("all")
