"""Tests the math module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
import pytest
from scipy import stats

from utilitiez import geomspace_int, random_uniform_fixed_sum, xlogx


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


def test_geomspace_int():
    """Test the `geomspace_int` function."""
    for num in [3, 20]:
        for a, b in [[0, 5], [1, 100]]:
            x = geomspace_int(a, b, num)
            assert np.issubdtype(x.dtype, np.integer)
            assert x[0] == a
            assert x[-1] == b
            assert len(x) <= num

    x = geomspace_int(10, 1000, 32)
    y = np.geomspace(10, 1000, 32)
    np.testing.assert_allclose(x - y, 0, atol=1)

    assert np.issubdtype(geomspace_int(0, 1, 0).dtype, np.integer)
    assert np.issubdtype(geomspace_int(0, 0, 10).dtype, np.integer)
    np.testing.assert_equal(geomspace_int(0, 1, 0), np.array([]))
    np.testing.assert_equal(geomspace_int(0, 0, 10), np.array([0]))
    np.testing.assert_equal(geomspace_int(0, 10, 1), np.array([0]))
    np.testing.assert_equal(geomspace_int(0, 2, 10), np.array([0, 1, 2]))
    np.testing.assert_equal(geomspace_int(0, 20, 2), np.array([0, 20]))
    np.testing.assert_equal(geomspace_int(0, 20, 3), np.array([0, 1, 20]))

    x = geomspace_int(10, 100, 20)
    y = geomspace_int(100, 10, 20)
    np.testing.assert_equal(x, y[::-1])

    with pytest.raises(ValueError):
        geomspace_int(0, 1, -1)
    with pytest.raises(ValueError):
        geomspace_int(-1, 2)
    with pytest.raises(ValueError):
        geomspace_int(1, -2)
