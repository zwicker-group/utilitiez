"""Tests the math module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
import pytest
from scipy import optimize

from utilitiez import xlogx


@pytest.mark.parametrize("jit", [True, False])
def test_xlogx(jit):
    """Test xlogx function."""
    # prepare function
    if jit:

        @nb.njit
        def f(value):
            return xlogx(value)
    else:
        f = xlogx

    xs = np.array([-1, 0, 0.5, 1])
    ys = np.array([np.nan, 0, 0.5 * np.log(0.5), 0])

    # test scalar data
    for x, y in zip(xs, ys, strict=False):
        if np.isnan(y):
            assert np.isnan(f(x))
        else:
            assert f(x) == pytest.approx(y)

    # test 1d arrays
    np.testing.assert_almost_equal(f(xs), ys)

    # test 2d arrays
    np.testing.assert_almost_equal(f(np.c_[xs, xs]), np.c_[ys, ys])


@pytest.mark.parametrize("threshold", [0, 1e-4, 0.1])
def test_xlogx_numpy(threshold):
    """Test the xlogx function and connected functions."""
    if threshold == 0:
        assert xlogx(threshold, threshold) == 0
    else:
        assert xlogx(threshold, threshold) == threshold * np.log(threshold)
        x = threshold - 1e-8
        assert xlogx(x, threshold) == pytest.approx(x * np.log(x))
    x = threshold + 1e-8
    assert xlogx(x, threshold) == pytest.approx(x * np.log(x))

    # test derivative
    if threshold > 0:
        x = threshold + np.array([-1e-6, 0, 1e-6])
    else:
        x = 1e-6
    xDiff = optimize.approx_fprime(x, lambda val: xlogx(val, threshold), epsilon=1e-10)
    np.testing.assert_allclose(np.diag(xDiff), xlogx(x, threshold, diff=1), rtol=1e-5)

    # test second derivative
    xDiff2 = optimize.approx_fprime(
        x, lambda val: xlogx(val, threshold, diff=1), epsilon=1e-10
    )
    np.testing.assert_allclose(np.diag(xDiff2), xlogx(x, threshold, diff=2), rtol=1e-4)

    for val in [-1, np.array([-0.5, 0.5])]:
        for diff in [None, 0, 1]:
            if threshold == 0:
                with pytest.raises(ValueError):
                    xlogx(val, diff=diff, threshold=threshold, raise_error=True)
            else:
                xlogx(val, diff=diff, threshold=threshold, raise_error=True)
            xlogx(val, diff=diff, threshold=threshold, raise_error=False)


@pytest.mark.parametrize("threshold", [0, 1e-4, 0.1])
@pytest.mark.parametrize("diff", [None, 0, 1, 2])
def test_xlogx_numba(threshold, diff):
    """Test the xlogx function and connected functions."""

    @nb.njit
    def get_value(x):
        return xlogx(x, threshold=threshold, diff=diff)

    assert xlogx(0, threshold=threshold, diff=diff) == get_value(0)
    assert xlogx(1e-5, threshold=threshold, diff=diff) == get_value(1e-5)
    assert xlogx(1e-3, threshold=threshold, diff=diff) == get_value(1e-3)
    assert xlogx(0.5, threshold=threshold, diff=diff) == get_value(0.5)
    x = np.array([0, 1e-5, 1e-3, 0.5])
    np.testing.assert_allclose(xlogx(x, threshold=threshold, diff=diff), get_value(x))

    for val in [-1, np.array([-0.5, 0.5])]:
        if threshold == 0 and diff != 2:
            with pytest.raises(ValueError):
                xlogx(val, diff=diff, threshold=threshold, raise_error=True)
        else:
            xlogx(val, diff=diff, threshold=threshold, raise_error=True)
        xlogx(val, diff=diff, threshold=threshold, raise_error=False)
