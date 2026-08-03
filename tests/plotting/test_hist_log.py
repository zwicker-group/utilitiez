"""Tests the densityplot module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison

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


@image_comparison(["hist_log_range"], extensions=["png", "pdf"], style="mpl20", tol=6)
def test_hist_range():
    """Test setting range in `hist_log`."""
    data = np.linspace(1, 100, 100)
    hist_log(data, range=[10, 100])


@image_comparison(
    ["hist_log_negative"], extensions=["png", "pdf"], style="mpl20", tol=6
)
def test_hist_negative():
    """Test negative values in `hist_log`."""
    data = np.linspace(-1, 1, 100)
    hist_log(data)


@image_comparison(["hist_double_log"], extensions=["png", "pdf"], style="mpl20", tol=6)
def test_hist_double_log():
    """Test double logarithmic plot."""
    data = np.linspace(1, 100, 100)
    hist_log(data, xlog=True, ylog=True)


@check_figures_equal(extensions=["png", "pdf"], tol=6)
def test_no_log(fig_test, fig_ref):
    """Test plot without logarithmic scaling."""
    data = np.linspace(1, 100, 100)
    hist_log(data, xlog=False, ylog=False, ax=fig_test.gca())
    fig_ref.gca().hist(data)


def test_no_log_arg():
    """Test that log arg cannot be used."""
    data = np.linspace(1, 100, 100)
    with pytest.raises(ValueError):
        hist_log(data, log=True)
