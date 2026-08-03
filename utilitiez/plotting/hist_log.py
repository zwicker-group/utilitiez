"""Helper functions for making histograms with log-scaled data.

.. autosummary::
   :nosignatures:

   ~hist_log

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def hist_log(
    data: ArrayLike,
    bins: int | str | ArrayLike = 10,
    range: tuple[float, float] | None = None,  # noqa: A002
    *,
    ax=None,
    xlog: bool = True,
    ylog: bool = False,
    **kwargs,
):
    """Plots a histogram with logarithmic bin widths.

    Args:
        data (array-like):
            The (positive) data to be plotted as a histogram
        bins (int, str, list of floats):
            If bins is an integer, it defines the number of bins in the range. Bins will
            have equal width in logarithmic space. If bins is a sequence, it defines the
            bin edges, including the left edge of the first bin and the right edge of
            the last bin; in this case, bins may be unequally spaced. If bins is a
            string, it is one of the binning strategies supported by
            :func:`numpy.histogram_bin_edges`.
        range (tuple or None):
            The lower and upper range of the bins. Lower and upper outliers are ignored.
            If not provided, range is ``(x.min(), x.max())``. Range has no effect if
            bins is a sequence. If bins is a sequence or range is specified, autoscaling
            is based on the specified bin range instead of the range of x.
        ax (:class:`matplotlib.axes.Axes`, optional):
            The axes on which to plot. If None, the current axes will be used.
        xlog (bool):
            If False, the bins will be chosen linearly.
        ylog (bool):
            If True, the histogram y-axis will be set to a log scale.
        **kwargs:
            Additional parameters are forwarded to :func:`matplotlib.hist`

    Returns:
        :class:`matplotlib.patches.StepPatch`
            The path object representing the histogram.
    """
    # extract the positive part of the data
    data = np.atleast_1d(data)
    data_pos = data[data > 0]
    if len(data_pos) < len(data):
        warnings.warn("Negative values are ignored", stacklevel=2)

    if isinstance(bins, (int, str)):
        if xlog:
            # determine logarithmically spaced bin edges automatically
            if range is None:
                log_range = None
            else:
                log_range = (np.log(range[0]), np.log(range[1]))
            bin_edges_log = np.histogram_bin_edges(
                np.log(data_pos), bins=bins, range=log_range
            )
            bin_edges = np.exp(bin_edges_log)
        else:
            # determine uniformly spaced bin edges automatically
            bin_edges = np.histogram_bin_edges(data_pos, bins=bins, range=range)
    else:
        bin_edges = bins

    # set the y-scale correctly
    if "log" in kwargs:
        msg = "Argument `log` is ambigious. Use `xlog` or `ylog` instead."
        raise ValueError(msg)
    kwargs["log"] = ylog

    # determine axes used for plotting
    plt_ax = plt.gca() if ax is None else ax

    # do the plot and set the scale of the y-axis
    plot = plt_ax.hist(data, bins=bin_edges, **kwargs)
    if xlog:
        plt_ax.set_xscale("log")
    return plot
