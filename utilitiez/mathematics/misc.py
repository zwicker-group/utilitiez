"""Mathematical functions.

.. autosummary::
   :nosignatures:

   ~geomspace_int
   ~random_uniform_fixed_sum

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable

if TYPE_CHECKING:
    from numpy.typing import NDArray


def geomspace_int(
    start: int, end: int, num: int = 50, *, max_steps: int = 100
) -> NDArray[np.integer]:
    """Return integers spaced (approximately) evenly on a log scale.

    Parameters:
        start (int):
            The starting value of the sequence.
        end (int):
            The final value of the sequence.
        num (int, optional):
            Number of samples to generate. Default is 50.
        max_steps (int, optional)
            The maximal number of steps of the iterative algorithm. If the algorithm
            could not find a solution, a `RuntimeError` is raised.

    Returns:
        an ordered sequence of at most `num` integers from `start` to `end` with
        approximately logarithmic spacing.
    """
    # check whether the supplied number is valid
    num = int(num)
    if num < 0:
        msg = f"Number of samples, {num}, must be non-negative."
        raise ValueError(msg)
    if num == 0:
        return np.array([], dtype=int)

    # check corner cases
    start = int(start)
    end = int(end)
    if start < 0 or end < 0:
        msg = "`start` and `end` must be positive numbers"
        raise ValueError(msg)
    if num == 1 or start == end:
        return np.array([start])

    if start > end:
        # inverted sequence
        return geomspace_int(end, start, num)[::-1]

    if num == 2:
        # return end intervals, which could be inverted by above line
        return np.array([start, end])

    if num > end - start:
        # all integers need to be returned
        return np.arange(start, end + 1)

    # calculate the maximal size of underlying logarithmic range
    if start == 0:
        start = 1
        num -= 1
        add_zero = True
    else:
        add_zero = False

    num_max = int(
        np.ceil((math.log(end) - math.log(start)) / (math.log(end) - math.log(end - 1)))
    )
    a, b = num, num_max  # interval of log-range
    n = a

    # try different log-ranges
    for _ in range(max_steps):
        # determine discretized logarithmic range
        ys_float = np.geomspace(start, end, num=n)
        ys = np.unique(ys_float.astype(int))
        ys_len = len(ys)

        if ys_len == num:
            break  # reached correct number

        if ys_len < num:
            # n is too small
            a = n
            n = int(math.sqrt(n * b))
            if a == n:
                n += 1
                if n == b:
                    break

        elif ys_len > num:
            # n is too large
            b = n
            n = int(math.sqrt(a * n))
            if b == n:
                n -= 1
                if n == a:
                    break
    else:
        msg = "Exceeded attempts"
        raise RuntimeError(msg)

    if add_zero:
        return np.r_[0, ys]  # type: ignore
    return ys


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
        x = np.random.beta(1, dim - d - 1) * x_max  # noqa: NPY002
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
        Array of shape (size, dim) with `size` samples of `dim` positive values adding
        to 1
    """
    if size < 1:
        msg = "size must be positive integer"
        raise ValueError(msg)

    xs: np.ndarray[Any, np.dtype[np.double]] = np.empty((size, dim))
    x_max = np.ones(size)
    for d in range(dim - 1):
        x = np.random.beta(1, dim - d - 1, size) * x_max  # noqa: NPY002
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

    if isinstance(size, int):
        # returns a 2d array of shape (size, dim)
        return _random_uniform_fixed_sum_multiple_samples(dim, size)

    msg = "`size` must be a positive integer or None"
    raise TypeError(msg)


@overload(random_uniform_fixed_sum)
def random_uniform_fixed_sum_ol(dim, size=None):
    """Overload `random_uniform_fixed_sum` to allow using it from numba code."""
    if not isinstance(dim, nb.types.Integer):
        msg = "`dim` must be an integer"
        raise nb.TypingError(msg)

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
        msg = "`size` must be positive integer or None"
        raise nb.TypingError(msg)

    return impl


__all__ = ["geomspace_int", "random_uniform_fixed_sum"]
