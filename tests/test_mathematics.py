"""Tests the math module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
import pytest

from utilitiez import xlogx


@pytest.mark.parametrize("jit", [True, False])
def test_xlogx(jit):
    """Test xlogx function with scalar values."""
    if jit:

        @nb.njit
        def f(x):
            return xlogx(x)
    else:
        f = xlogx

    xs = np.array([-1, 0, 0.5, 1])
    ys = np.array([np.nan, 0, 0.5 * np.log(0.5), 0])

    # test scalar data
    for x, y in zip(xs, ys):
        if np.isnan(y):
            assert np.isnan(f(x))
        else:
            assert f(x) == pytest.approx(y)

    # test 1d arrays
    np.testing.assert_almost_equal(f(xs), ys)

    # test 2d arrays
    np.testing.assert_almost_equal(f(np.c_[xs, xs]), np.c_[ys, ys])
