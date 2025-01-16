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
    f = do_jit(random_uniform_fixed_sum, jit)
    xs = f(dim)
    assert xs.shape == (dim,)
    assert xs.sum() == pytest.approx(1)


@pytest.mark.parametrize("jit", [True, False])
def test_random_uniform_fixed_sum_dist(jit):
    """Test distribution of get_uniform_random_composition."""
    f = do_jit(random_uniform_fixed_sum, jit)
    xs = np.array([f(3) for _ in range(1000)])
    # test that all variables have similar
    assert stats.ks_2samp(xs[:, 0], xs[:, 1]).pvalue > 0.05
    assert stats.ks_2samp(xs[:, 0], xs[:, 2]).pvalue > 0.05
    assert stats.ks_2samp(xs[:, 1], xs[:, 2]).pvalue > 0.05
