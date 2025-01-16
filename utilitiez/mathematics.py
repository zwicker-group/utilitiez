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
        return xlogx_scalar(x)  # type: ignore

    x = np.asanyarray(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = x * np.log(x)
    y[x == 0] = 0
    return y  # type: ignore


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


@register_jitable
def random_uniform_fixed_sum(dim: int) -> np.ndarray:
    """Sample uniformly distributed positive random numbers adding to 1.

    Args:
        dim (int): the number of values to return

    Returns:
        An array with `dim` random positive fractions that add to 1
    """
    xs = np.empty(dim)
    x_max = 1.0
    for d in range(dim - 1):
        x = np.random.beta(1, dim - d - 1) * x_max
        x_max -= x
        xs[d] = x
    xs[-1] = 1 - xs[:-1].sum()
    return xs
