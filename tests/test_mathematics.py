"""Tests the math module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
import pytest
from scipy import stats

from utilitiez import random_uniform_fixed_sum, xlogx


def do_jit(func, jit=True):
    """Return a jitted version of a function."""
    if jit:

        @nb.njit
        def f(x):
            return func(x)
    else:
        f = func

    return f


@pytest.mark.parametrize("jit", [True, False])
def test_xlogx(jit):
    """Test xlogx function with scalar values."""
    f = do_jit(xlogx, jit)

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


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_random_uniform_fixed_sum(dim, jit):
    """Test get_uniform_random_composition function."""
    # get some samples
    f = do_jit(random_uniform_fixed_sum, jit)
    xs = np.array([f(dim) for _ in range(10000)])

    # check basic properties
    assert xs.shape == (10000, dim)
    assert np.allclose(xs.sum(axis=1), 1)

    # check the distributions agains the expectations
    if dim == 1:
        np.testing.assert_allclose(xs, 1)
    elif dim == 2:
        cdf = stats.uniform.cdf
        assert stats.ks_1samp(xs[:, 0], cdf).statistic < 0.1
        assert stats.ks_1samp(xs[:, 1], cdf).statistic < 0.1
    elif dim == 3:
        cdf = stats.triang(0).cdf
        assert stats.ks_1samp(xs[:, 0], cdf).statistic < 0.1
        assert stats.ks_1samp(xs[:, 1], cdf).statistic < 0.1
        assert stats.ks_1samp(xs[:, 2], cdf).statistic < 0.1
    else:
        raise NotImplementedError("Check not implemented for dim>3")
