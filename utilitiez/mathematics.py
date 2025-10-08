"""Mathematical functions.

.. autosummary::
   :nosignatures:

   ~random_uniform_fixed_sum
   ~xlogx

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from typing import Any

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable
from numpy.typing import ArrayLike


def xlogx_scalar(x):
    r"""Calculates :math:`x \log(x)`, including the corner case x == 0.

    Args:
        x (float): The argument

    Returns:
        float: The result
    """
    if x < 0:
        return math.nan
    elif x == 0:
        return 0
    else:
        return x * np.log(x)


def xlogx(x: ArrayLike | float) -> ArrayLike | float:
    r"""Calculates :math:`x \log(x)`, including the corner case x == 0.

    Args:
        x (array or float): The argument

    Returns:
        array or float: The result
    """
    if np.isscalar(x):
        return xlogx_scalar(x)

    x = np.asanyarray(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = x * np.log(x)
    y[x == 0] = 0
    return y


@overload(xlogx)
def xlogx_ol(x):
    """Overload `xlogx` to allow using it from numba code."""
    if isinstance(x, nb.types.Number):
        return xlogx_scalar

    elif isinstance(x, nb.types.Array):
        xlogx_single = nb.njit(xlogx_scalar)

        def xlogx_impl(x):
            y = np.empty_like(x)
            for i in range(x.size):
                y.flat[i] = xlogx_single(x.flat[i])
            return y

        return xlogx_impl

    else:
        raise nb.TypingError("Only accepts numbers or NumPy ndarray")


def _random_uniform_fixed_sum_single_sample(
    dim: int,
) -> np.ndarray[Any, np.dtype[np.double]]:
    """Sample uniformly distributed positive random numbers adding to 1.

    Args:
        dim (int): the number of values to return

    Returns:
        An array with `dim` random positive fractions that add to 1
    """
    xs: np.ndarray[Any, np.dtype[np.double]] = np.empty(dim)
    x_max = 1.0
    for d in range(dim - 1):
        x = np.random.beta(1, dim - d - 1) * x_max
        x_max -= x
        xs[d] = x
    xs[-1] = 1 - xs[:-1].sum()
    return xs


def _random_uniform_fixed_sum_multiple_samples(
    dim: int, size: int
) -> np.ndarray[Any, np.dtype[np.double]]:
    """Sample uniformly distributed positive random numbers adding to 1.

    Args:
        dim (int): the number of values to return
        size (int): the number of samples to return

    Returns:
        Array of shape (size, dim) with `size` samples of `dim` positive values adding to 1
    """
    if size < 1:
        raise ValueError("size must be positive integer")

    xs: np.ndarray[Any, np.dtype[np.double]] = np.empty((size, dim))
    x_max = np.ones(size)
    for d in range(dim - 1):
        x = np.random.beta(1, dim - d - 1, size) * x_max
        x_max -= x
        xs[:, d] = x
    xs[:, -1] = 1 - xs[:, :-1].sum(axis=1)
    return xs


def random_uniform_fixed_sum(
    dim: int, size: int | None = None
) -> np.ndarray[Any, np.dtype[np.double]]:
    """Sample uniformly distributed positive random numbers adding to 1.

    Args:
        dim (int):
            The number of values that sum to 1
        size (int):
            The number of independent samples to return. If size is `None` a 1d array of
            numbers is returned. If size is a number the returned array has shape
            `(size, dim)`.

    Returns:
        If `size` is `None`, returns a 1D array of shape `(dim,)` containing positive
            values that sum to 1.
        If `size` is an integer, returns a 2D array of shape `(size, dim)` where each
            row contains positive values that sum to 1.
    """
    if size is None:
        # returns a 1d array of shape (dim)
        return _random_uniform_fixed_sum_single_sample(dim)

    elif isinstance(size, int):
        # returns a 2d array of shape (size, dim)
        return _random_uniform_fixed_sum_multiple_samples(dim, size)

    else:
        raise TypeError("`size` must be a positive integer or None")


@overload(random_uniform_fixed_sum)
def random_uniform_fixed_sum_ol(dim, size=None):
    """Overload `random_uniform_fixed_sum` to allow using it from numba code."""
    if not isinstance(dim, nb.types.Integer):
        raise nb.TypingError("`dim` must be an integer")

    # compile the function getting a single sample, which we need in all cases
    single_sample = register_jitable(_random_uniform_fixed_sum_single_sample)

    if size is None or isinstance(size, nb.types.NoneType):
        # return compiled version of a single sample

        def impl(dim, size=None):
            return single_sample(dim)

    elif isinstance(size, nb.types.Integer):
        # return compiled version making many samples

        def impl(dim, size=None):
            out = np.empty((size, dim))
            for i in nb.prange(size):
                out[i, :] = single_sample(dim)
            return out

    else:
        raise nb.TypingError("`size` must be positive integer or None")

    return impl
