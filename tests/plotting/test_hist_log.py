"""Tests the densityplot module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
from matplotlib.testing.decorators import image_comparison

from utilitiez import hist_log


@image_comparison(["simple_hist_log"], extensions=["png", "pdf"], style="mpl20", tol=6)
def test_simple_hist_log():
    """Test simple usage similar to pyplot interface."""
    data = np.linspace(1, 100, 100)
    hist_log(data)


@image_comparison(["hist_log_bins"], extensions=["png", "pdf"], style="mpl20", tol=6)
def test_hist_bins():
    """Test setting bins in `hist_log`."""
    data = np.linspace(1, 100, 100)
    bins = [1, 20, 50, 100]
    hist_log(data, bins=bins)


@image_comparison(
    ["hist_log_negative"], extensions=["png", "pdf"], style="mpl20", tol=6
)
def test_hist_negative():
    """Test negative values in `hist_log`."""
    data = np.linspace(-1, 1, 100)
    hist_log(data)
