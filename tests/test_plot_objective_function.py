import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("agg")

from gradient_free_optimization_plots import plot_objective_function_2d


class TestPlotObjectiveFunction2d:
    def test_returns_fig_and_ax(self, sphere_function, search_space):
        fig, ax = plot_objective_function_2d(sphere_function, search_space)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_figsize(self, sphere_function, search_space):
        fig, ax = plot_objective_function_2d(
            sphere_function, search_space, figsize=(10, 10)
        )
        width, height = fig.get_size_inches()
        assert width == 10
        assert height == 10
        plt.close("all")

    def test_custom_alpha(self, sphere_function, search_space):
        fig, ax = plot_objective_function_2d(
            sphere_function, search_space, alpha=0.5
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_has_image_on_axes(self, sphere_function, search_space):
        fig, ax = plot_objective_function_2d(sphere_function, search_space)
        assert len(ax.images) > 0
        plt.close("all")
