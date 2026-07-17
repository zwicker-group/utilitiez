"""Helper functions for making histograms with log-scaled data.

.. autosummary::
   :nosignatures:

   ~hist_log

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from numpy.typing import ArrayLike


def hist_log(
    data: ArrayLike,
    bins: int | str | ArrayLike = 10,
    range: tuple[float, float] | None = None,
    *,
    ax=None,
    **kwargs,
) -> mpatches.StepPatch:
    """Plots a histogram with logarithmic bin widths."""
    # extract the positive part of the data
    data = np.atleast_1d(data)
    data_pos = data[data > 0]
    if len(data_pos) < len(data):
        warnings.warn("Negative values are ignored", stacklevel=2)

    if isinstance(bins, (int, str)):
        bin_edges = np.histogram_bin_edges(np.log(data_pos), bins=bins, range=range)
    else:
        bin_edges = bins

    # determine axes used for plotting
    if ax is None:
        plt_ax = plt.gca()
    else:
        plt_ax = ax

    plot = plt_ax.hist(data, bins=bin_edges, **kwargs)
    plt_ax.set_xscale("log")
    return plot
