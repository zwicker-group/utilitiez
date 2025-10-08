"""Tests the math module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
import pytest
from scipy import stats

from utilitiez import random_uniform_fixed_sum, xlogx


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
    for x, y in zip(xs, ys):
        if np.isnan(y):
            assert np.isnan(f(x))
        else:
            assert f(x) == pytest.approx(y)

    # test 1d arrays
    np.testing.assert_almost_equal(f(xs), ys)

    # test 2d arrays
    np.testing.assert_almost_equal(f(np.c_[xs, xs]), np.c_[ys, ys])


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_random_uniform_fixed_sum_single_sample(dim, jit):
    """Test random_uniform_fixed_sum function for single samples."""
    # prepare function
    if jit:

        @nb.njit
        def f(dim):
            return random_uniform_fixed_sum(dim)
    else:
        f = random_uniform_fixed_sum

    # get some samples
    xs = np.array([f(dim) for _ in range(10_000)])

    # check basic properties
    assert xs.shape == (10_000, dim)
    assert np.allclose(xs.sum(axis=1), 1)

    # check the distributions against the expectations
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


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_random_uniform_fixed_sum_multiple_sample(dim, jit):
    """Test random_uniform_fixed_sum function for multiple samples."""
    # prepare function
    if jit:

        @nb.njit
        def f(dim, size):
            return random_uniform_fixed_sum(dim, size)
    else:
        f = random_uniform_fixed_sum

    # simple test case
    assert f(3, None).shape == (3,)
    with pytest.raises(ValueError):
        f(3, -1)
    with pytest.raises((TypeError, nb.TypingError)):
        f(3, "wrong")

    # get some samples
    xs = f(dim, size=10_000)

    # check basic properties
    assert xs.shape == (10_000, dim)
    assert np.allclose(xs.sum(axis=1), 1)

    # check the distributions against the expectations
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
