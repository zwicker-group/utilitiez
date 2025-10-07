"""Mathematical functions.

.. autosummary::
   :nosignatures:

   ~random_uniform_fixed_sum
   ~xlogx


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from typing import Callable

import numba as nb
import numpy as np
from numba.core.errors import TypingError
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
        raise TypingError("Only accepts numbers or NumPy ndarray")



def random_uniform_fixed_sum_multiple_samples(dim: int, size: int) -> np.ndarray:
    """Sample uniformly distributed positive random numbers adding to 1.

    Args:
        dim (int): the number of values to return
        size (int): the number of samples to return
       
    Returns:
        An array of shape (size, dim). It contains `size` samples of arrays of `dim` random positive fractions that add to 1
    """
    assert size > 1, "size must be > 1. Otherwise use random_uniform_fixed_sum_single_sample() instead"

    xs: np.ndarray = np.empty((size, dim))
    x_max = np.ones(size)
    for d in range(dim - 1):
        x = np.random.beta(1, dim - d - 1, size) * x_max
        x_max -= x
        xs[:, d] = x
    xs[:, -1] = 1 - xs[:, :-1].sum(axis=1)
   
    return xs

def random_uniform_fixed_sum_single_sample(dim: int) -> np.ndarray:
    """Sample uniformly distributed positive random numbers adding to 1.

    Args:
        dim (int): the number of values to return

    Returns:
        An array with `dim` random positive fractions that add to 1
    """
    xs: np.ndarray = np.empty(dim)
    x_max = 1.0
    for d in range(dim - 1):
        x = np.random.beta(1, dim - d - 1) * x_max
        x_max -= x
        xs[d] = x
    xs[-1] = 1 - xs[:-1].sum()
    return xs


def random_uniform_fixed_sum(dim: int, size: int) -> np.ndarray:
    """Sample uniformly distributed positive random numbers adding to 1.
    Args:
        dim (int): the number of values to return
        size (int): the number of samples to return. If size=1, a 1d array is returned, otherwise a 2d array of shape (size, dim)
    """
    if size == 1:
        #returns a 1d array of shape (dim)
        return random_uniform_fixed_sum_single_sample(dim)
    else:
        #returns a 2d array of shape (size, dim)
        return random_uniform_fixed_sum_multiple_samples(dim, size)

@overload(random_uniform_fixed_sum)
# ToDo: implement a numba version