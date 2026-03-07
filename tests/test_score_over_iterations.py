import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("agg")

from gradient_free_optimizers import HillClimbingOptimizer, RandomSearchOptimizer

from gradient_free_optimization_plots import score_over_iter_plot
from gradient_free_optimization_plots.score_over_iterations import get_best_scores_iter


class TestGetBestScoresIter:
    def test_monotonically_increasing(self):
        search_data = pd.DataFrame({"score": [1.0, 0.5, 2.0, 1.5, 3.0]})
        result = get_best_scores_iter(search_data)
        scores = list(result["score"])
        assert scores == [1.0, 1.0, 2.0, 2.0, 3.0]

    def test_already_sorted(self):
        search_data = pd.DataFrame({"score": [1.0, 2.0, 3.0]})
        result = get_best_scores_iter(search_data)
        scores = list(result["score"])
        assert scores == [1.0, 2.0, 3.0]

    def test_all_same(self):
        search_data = pd.DataFrame({"score": [5.0, 5.0, 5.0]})
        result = get_best_scores_iter(search_data)
        scores = list(result["score"])
        assert scores == [5.0, 5.0, 5.0]

    def test_single_value(self):
        search_data = pd.DataFrame({"score": [42.0]})
        result = get_best_scores_iter(search_data)
        assert list(result["score"]) == [42.0]

    def test_returns_dataframe(self):
        search_data = pd.DataFrame({"score": [1.0, 2.0]})
        result = get_best_scores_iter(search_data)
        assert isinstance(result, pd.DataFrame)
        assert "score" in result.columns


class TestScoreOverIterPlot:
    def test_returns_fig_and_ax(self, sphere_function, search_space):
        fig, ax = score_over_iter_plot(
            optimizer_l=[HillClimbingOptimizer],
            n_iter=10,
            n_runs=2,
            objective_function=sphere_function,
            search_space=search_space,
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_multiple_optimizers(self, sphere_function, search_space):
        fig, ax = score_over_iter_plot(
            optimizer_l=[HillClimbingOptimizer, RandomSearchOptimizer],
            n_iter=10,
            n_runs=2,
            objective_function=sphere_function,
            search_space=search_space,
        )
        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close("all")

    def test_legend_present(self, sphere_function, search_space):
        fig, ax = score_over_iter_plot(
            optimizer_l=[HillClimbingOptimizer],
            n_iter=10,
            n_runs=2,
            objective_function=sphere_function,
            search_space=search_space,
        )
        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")
