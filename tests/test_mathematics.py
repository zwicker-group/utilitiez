"""Tests the math module.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numba as nb
import numpy as np
import pytest
from scipy import stats
import os
import importlib.util

from utilitiez import random_uniform_fixed_sum, xlogx



def do_jit(func, jit=True):
    """Return a jitted version of a function."""
    if jit:
        jitted_func = nb.njit(func)
        return jitted_func
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
@pytest.mark.parametrize("size", [1, 2, 3])
def test_random_uniform_fixed_sum(dim, jit, size, squeeze):
    """Test get_uniform_random_composition function."""

    f = do_jit(random_uniform_fixed_sum, jit)
    N = 10000

    def check_distrs(xs, dim, sample_idx):
        assert xs.ndim == 3, "Expected 3d array"
        assert xs.shape[1] > sample_idx, "Data does not contain required length of samples"

        if dim == 1:
            np.testing.assert_allclose(xs, 1)
        elif dim == 2:
            cdf = stats.uniform.cdf
            assert stats.ks_1samp(xs[:, sample_idx, 0], cdf).statistic < 0.1
            assert stats.ks_1samp(xs[:, sample_idx, 1], cdf).statistic < 0.1
        elif dim == 3:
            cdf = stats.triang(0).cdf
            assert stats.ks_1samp(xs[:, sample_idx, 0], cdf).statistic < 0.1
            assert stats.ks_1samp(xs[:, sample_idx, 1], cdf).statistic < 0.1
            assert stats.ks_1samp(xs[:, sample_idx, 2], cdf).statistic < 0.1
        else:
            raise NotImplementedError("Check not implemented for dim>3")

   
    # get some samples
    xs_as_before = np.array([f(dim) for _ in range(N)]) #call to old function signature
    xs_multi_sample = np.array([f(dim, size) for _ in range(N)]) #call to multiple-sample signature

    #Check single sample signature call
    assert xs_as_before.ndim == 2
    assert xs_as_before.shape == (N, dim)
    assert np.allclose(xs_as_before.sum(axis=-1), 1)
    check_distrs(xs_as_before[:, None, :], dim, 0)  # add dummy sample axis

    #Check multiple sample signature call
    assert xs_multi_sample.ndim == 3
    assert xs_multi_sample.shape == (N, size, dim)
    assert np.allclose(xs_multi_sample.sum(axis=-1), 1)
    for sample_idx in range(size):
        check_distrs(xs_multi_sample, dim, sample_idx)

