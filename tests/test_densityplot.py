"""Tests the densityplot module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.testing.decorators import image_comparison

from utilitiez import densityplot


def test_most_basic(rng):
    """Test whether we can simply call the method with just data."""
    data = rng.uniform(size=(4, 3))
    densityplot(data)


@image_comparison(["simple_pyplot"], extensions=["png", "pdf"], tol=6)
def test_simple_pyplot():
    """Test simple usage similar to pyplot interface."""
    x = np.linspace(1, 1e2, 6)
    y = np.geomspace(1e1, 1e5, 4)
    data = np.sin(x[:, np.newaxis] * y[np.newaxis, :])

    # this would be a standard usage that we envision
    densityplot(data, x, y)
    plt.colorbar()
    plt.xticks([1, 10, 50], [1, 10, 50])
    plt.ylabel("y-axis")


@image_comparison(["simple_objects"], extensions=["png", "pdf"], tol=6)
def test_simple_objects():
    """Test simple usage using the object-oriented interface."""
    x = np.geomspace(1, 1e2, 6)
    y = np.geomspace(1e1, 1e5, 4)
    data = np.sin(x[:, np.newaxis] * y[np.newaxis, :])

    # this would be a standard usage that we envision
    ax = plt.gca()
    im = densityplot(data, x, y, ax=ax)
    plt.colorbar(im, ax=ax)
    ax.set_aspect(0.5)
    ax.set_xticks([1, 10, 50])
    ax.set_xticklabels([1, 10, 50])
    ax.set_ylabel("y-axis")


@image_comparison(["linear"], extensions=["png", "pdf"], tol=6)
def test_linear():
    """Test density plots with linearly scaled axes."""
    x = np.linspace(0, 3, 6)
    y = np.linspace(3, 7, 4)
    data = np.sin(x[:, np.newaxis] * y[np.newaxis, :])

    densityplot(data, x, y)
    plt.colorbar()


@image_comparison(["loglog"], extensions=["png", "pdf"], tol=6)
def test_loglog():
    """Test density plots with logarithmically scaled axes."""
    x = np.geomspace(1, 1e2, 6)
    y = np.geomspace(1e1, 1e5, 4)
    data = np.sin(x[:, np.newaxis] * y[np.newaxis, :])

    densityplot(data, x, y)
    plt.colorbar()


@image_comparison(["loglin"], extensions=["png", "pdf"], tol=6)
def test_loglin():
    """Test density plots with axes with mixed scaling."""
    x = np.geomspace(1, 1e2, 6)
    y = np.linspace(3, 7, 4)
    data = np.sin(x[:, np.newaxis] * y[np.newaxis, :])

    densityplot(data, x, y)
    plt.colorbar()


@image_comparison(["plot_extras"], extensions=["png", "pdf"], tol=6)
def test_plot_extras():
    """Test density plots with axes with additional settings."""
    x = np.geomspace(1, 1e2, 6)
    y = np.linspace(3, 7, 4)
    data = np.sin(x[:, np.newaxis] * y[np.newaxis, :])

    args = {"cmap": "Blues", "vmin": -2, "vmax": 4, "alpha": 0.5}
    densityplot(data, x, y, **args)
    plt.colorbar()
    plt.xlabel("X label")
    plt.ylabel("Y label")


@image_comparison(["logcolor"], extensions=["png", "pdf"], tol=6)
def test_logcolor():
    """Test density plots with axes with mixed scaling."""
    x = np.geomspace(1, 1e2, 6)
    y = np.linspace(3, 7, 4)
    data = x[:, np.newaxis] * y[np.newaxis, :]
    norm = colors.LogNorm()

    densityplot(data, x, y, norm=norm)
    plt.colorbar()


@image_comparison(["weird_scaling"], extensions=["png", "pdf"], tol=6)
def test_weird_scaling(rng):
    """Test whether density plots with weird scaling works."""
    x = np.array([1, 2, 4, 5])
    y = np.array([4, 5, 10])
    data = np.sin(x[:, np.newaxis] * y[np.newaxis, :])
    densityplot(data, x, y)
