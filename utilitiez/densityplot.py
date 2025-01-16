"""Helper functions for making density plots with matplotlib.

.. autosummary::
   :nosignatures:

   ~densityplot

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axis, collections, image, ticker
from numpy.typing import ArrayLike

ScaleType = Literal["linear", "log", "general"]


def get_scale(x: np.ndarray) -> tuple[ScaleType, np.ndarray]:
    """Determine the scale associated with numbers in an array.

    Args:
        x (ArrayLike):
            Positions along the axes where data points are specified

    Returns:
        tuple giving the scales and coordinates for quads
    """
    # check whether the data is scaled linearly
    x_d = x[1:] - x[:-1]
    if np.allclose(x_d, x_d.mean()):
        dx = x_d.mean() / 2
        quads = np.linspace(x.min() - dx, x.max() + dx, len(x) + 1)
        return "linear", quads

    # check whether the data is scaled logarithmically
    x_d = x[1:] / x[:-1]
    if np.allclose(x_d, x_d.mean()):
        dx = np.sqrt(x_d.mean())
        quads = np.geomspace(x.min() / dx, x.max() * dx, len(x) + 1)
        return "log", quads

    x_ = (x[1:] + x[:-1]) / 2  # get mid points
    quads = np.r_[2 * x[0] - x_[0], x_, 2 * x[-1] - x_[-1]]  # add outer points
    assert len(x_) + 1 == len(x) == len(quads) - 1
    return "general", quads


def set_axis_ticks(
    axis: axis.Axis, scale: ScaleType, x: np.ndarray, *, maxnum: int = 5
) -> None:
    """Set the major and minor ticks on a given axis.

    Args:
        axis (:class:`~matplotlib.axis.Axis`):
            The axis object to set the ticks on.
        scale (ScaleType):
            The scale type of the axis, e.g., 'linear' or 'log'.
        x (ArrayLike):
            The positions of the major ticks.
        maxnum (int, optional):
            Maximal length of `x`, so that all ticks are set. Default is 5.
    """
    if len(x) < maxnum:
        axis.set_major_locator(ticker.FixedLocator(list(x)))
        if scale == "log":
            axis.set_minor_locator(ticker.LogLocator(subs="auto"))
            axis.set_major_formatter(
                ticker.LogFormatterMathtext(minor_thresholds=(np.inf, np.inf))
            )


def densityplot(
    data: ArrayLike,
    x: ArrayLike | None = None,
    y: ArrayLike | None = None,
    *,
    ax=None,
    set_ticks: bool = True,
    **kwargs,
) -> collections.QuadMesh:
    """Creates a density plot of the given data.

    Note:
        This function rasterizes the output of `pcolormesh` by default to decrease the
        file size and avoid some problems with PDF readers. To have images with
        sufficient quality, set a high DPI value when saving, e.g., by using
        :code:`plt.savefig("fig.pdf", dpi=300)`.

    Args:
        data (ArrayLike):
            A 2D array representing the density values.
        x: (ArrayLike):
            A 1D array defining x-axis values where data was sampled
        y (ArrayLike):
            A 1D array defining y-axis values where data was sampled
        ax (:class:`matplotlib.axes.Axes`, optional):
            The axes on which to plot. If None, the current axes will be used.
        set_ticks ()
        **kwargs
            Additional keyword arguments to pass to `pcolormesh`

    Returns:
        :class:`matplotlib.collections.QuadMesh`
            The image or mesh object created by the plot.
    """
    # check image data
    data = np.asanyarray(data)
    if not data.ndim == 2:
        raise TypeError("`data` must be a 2d array")

    # check x-axis
    if x is None:
        x = np.arange(data.shape[0]) + 0.5
    else:
        x = np.asanyarray(x)
    if x.ndim != 1 or len(x) < 2:
        raise TypeError("`x` must be 1d array with at least 2 entries")

    # check y-axis
    if y is None:
        y = np.arange(data.shape[1]) + 0.5
    else:
        y = np.asanyarray(y)
    if y.ndim != 1 or len(y) < 2:
        raise TypeError("`y` must be 1d array with at least 2 entries")

    # check consistency of inputs
    if data.shape != (len(x), len(y)):
        raise ValueError("`data` must have shape (len(x), len(y))")

    # get the types of the axes and additional properties
    x_scale, x_quads = get_scale(x)
    y_scale, y_quads = get_scale(y)

    # determine axes used for plotting
    if ax is None:
        plt_ax = plt.gca()
    else:
        plt_ax = ax

    # produce the plot
    kwargs.setdefault("rasterized", True)  # improves file size and image quality
    kwargs.setdefault("shading", "flat")  # show actual data values
    im = plt_ax.pcolormesh(x_quads, y_quads, data.T, **kwargs)

    # scale the axes correctly
    plt_ax.set_xscale("linear" if x_scale == "general" else x_scale)
    plt_ax.set_yscale("linear" if y_scale == "general" else y_scale)
    plt_ax.set_xlim(x_quads[0], x_quads[-1])
    plt_ax.set_ylim(y_quads[0], y_quads[-1])
    if set_ticks:
        set_axis_ticks(plt_ax.xaxis, x_scale, x)
        set_axis_ticks(plt_ax.yaxis, y_scale, y)

    # set image as current image if no axes was given, which is important for adding
    # colorbars using the pyplot interface
    if ax is None:
        plt.sci(im)
        plt.sca(plt_ax)

    return im


__all__ = ["densityplot"]
