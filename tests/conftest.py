import numpy as np
import pytest


@pytest.fixture
def sphere_function():
    def _sphere(para):
        return -(para["x0"] ** 2 + para["x1"] ** 2)

    return _sphere


@pytest.fixture
def search_space():
    return {
        "x0": np.arange(-5, 5, 0.5),
        "x1": np.arange(-5, 5, 0.5),
    }
