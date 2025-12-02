"""Implements an extended version of the `xlogx` function.

.. autosummary::
   :nosignatures:

   ~xlogx

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from typing import Union
from typing import overload as type_overload

import numba as nb
import numpy as np
from numba.extending import overload as numba_overload
from numba.extending import register_jitable
from numpy.typing import ArrayLike, NDArray

NoneType = type(None)
FloatOrArray = Union[float, NDArray[np.floating]]


@register_jitable
def _xlogx_diff0(x, threshold=0, raise_error=False):
    """Evaluate xlogx function for one point."""
    if x > threshold:
        return x * np.log(x)
    if threshold == 0:
        # without linearization
        if x == 0:
            return 0.0
        if raise_error:
            msg = "xlogx expects non-negative values"
            raise ValueError(msg)
        return math.nan
    # with linearization
    log_threshold = np.log(threshold)
    return 0.5 * (x - threshold) * (x / threshold + 1) + x * log_threshold


@register_jitable
def _xlogx_diff1(x, threshold=0, raise_error=False):
    """Evaluate first derivative of xlogx function for one point."""
    if x > threshold:
        return 1 + np.log(x)
    if threshold == 0:
        # without linearization
        if x == 0:
            return -np.inf
        if raise_error:
            msg = "xlogx expects non-negative values"
            raise ValueError(msg)
        return math.nan
    # with linearization
    return x / threshold + np.log(threshold)


@register_jitable
def _xlogx_diff2(x, threshold=0, raise_error=False):
    """Evaluate second derivative of xlogx function for one point."""
    if x > threshold:
        return 1 / x
    if threshold == 0:
        return np.inf
    return 1 / threshold


@nb.njit
def _xlogx_diff0_array(x, threshold=0, raise_error=False):
    """Evaluate xlogx function for an array."""
    y = np.empty_like(x)
    for i in range(x.size):
        y.flat[i] = _xlogx_diff0(
            x.flat[i], threshold=threshold, raise_error=raise_error
        )
    return y


@nb.njit
def _xlogx_diff1_array(x, threshold=0, raise_error=False):
    """Evaluate first derivative of xlogx function for an array."""
    y = np.empty_like(x)
    for i in range(x.size):
        y.flat[i] = _xlogx_diff1(
            x.flat[i], threshold=threshold, raise_error=raise_error
        )
    return y


@nb.njit
def _xlogx_diff2_array(x, threshold=0, raise_error=False):
    """Evaluate second derivative of xlogx function for an array."""
    y = np.empty_like(x)
    for i in range(x.size):
        y.flat[i] = _xlogx_diff2(
            x.flat[i], threshold=threshold, raise_error=raise_error
        )
    return y


@type_overload
def xlogx(
    x: float, threshold: float = 0, diff: int | None = None, raise_error: bool = False
) -> float: ...


@type_overload
def xlogx(
    x: ArrayLike,
    threshold: float = 0,
    diff: int | None = None,
    raise_error: bool = False,
) -> NDArray[np.floating]: ...


def xlogx(
    x: FloatOrArray,
    threshold: float = 0,
    diff: int | None = None,
    raise_error: bool = False,
) -> FloatOrArray:
    r"""Calculate :math:`x \log(x)` and its derivatives.

    This function optionally uses a linear approximation for small :math:`x` to
    approximate the function for small negative values, which otherwise would return
    NaN or raise an error (depending on the flag `raise_error`).

    Args:
        x (float or :class:`~numpy.ndarray`):
            The value or array to which the function is applied
        threshold (float):
            Threshold below which the function will be linearized to extend the support
            of the function to negative numbers. Setting `threshold` to zero disables
            the linearization, so negative values raise an error or return NaN.
        diff (int):
            Degree of differentiation of the expression. The default value `None`
            corresponds to `diff=0`.
        raise_error (bool):
            If True, a ValueError is raised for non-positive values in case
            threshold is 0.

    Returns:
        float: The result
    """
    if np.isscalar(x):
        if diff == 0 or diff is None:
            return _xlogx_diff0(x, threshold, raise_error=raise_error)
        if diff == 1:
            return _xlogx_diff1(x, threshold, raise_error=raise_error)
        if diff == 2:
            return _xlogx_diff2(x, threshold, raise_error=raise_error)
        msg = "Only diff={0, 1, 2} is implemented"
        raise NotImplementedError(msg)

    if diff == 0 or diff is None:
        return _xlogx_diff0_array(x, threshold=threshold, raise_error=raise_error)
    if diff == 1:
        return _xlogx_diff1_array(x, threshold=threshold, raise_error=raise_error)
    if diff == 2:
        return _xlogx_diff2_array(x, threshold=threshold, raise_error=raise_error)
    msg = f"diff={diff} is not implemented"
    raise NotImplementedError(msg)


# overload the optimized numpy xlogx function
@numba_overload(xlogx)
def xlogx_ol(x, threshold=0, diff=None, raise_error=False):
    """Generator for a numba-compiled xlogx function supporting scalars and arrays."""
    if not isinstance(threshold, (int, float, nb.types.Integer, nb.types.Float)):
        msg = f"`threshold` must be a number, got {threshold.__class__}"
        raise nb.TypingError(msg)
    if isinstance(threshold, nb.types.Literal) and threshold.literal_value < 0:
        msg = "`threshold` must be a non-negative number"
        raise ValueError(msg)
    if not isinstance(diff, (nb.types.Literal, nb.types.NoneType, NoneType)):
        msg = f"`diff` must be a compile-time constant, not {diff.__class__}"
        raise nb.TypingError(msg)
    if not isinstance(diff, (int, nb.types.Integer, nb.types.NoneType, NoneType)):
        msg = f"`diff` must be an integer, not {diff.__class__}"
        raise nb.TypingError(msg)

    # determine degree of differentiation
    if diff is None or isinstance(diff, nb.types.NoneType):
        diff_val = 0
    elif isinstance(diff, nb.types.Literal):
        diff_val = diff.literal_value
    else:
        diff_val = diff

    if isinstance(x, nb.types.Number):
        # return an implementation for scalar values
        impl = [_xlogx_diff0, _xlogx_diff1, _xlogx_diff2][diff_val]

        def xlogx_scalar(x, threshold=0, diff=None, raise_error=False):
            return impl(x, threshold=threshold, raise_error=raise_error)

        return xlogx_scalar

    if isinstance(x, nb.types.Array):
        # return a vectorized implementation
        impl = [_xlogx_diff0_array, _xlogx_diff1_array, _xlogx_diff2_array][diff_val]

        def xlogx_array(x, threshold=0, diff=None, raise_error=False):
            return impl(x, threshold=threshold, raise_error=raise_error)

        return xlogx_array

    msg = "Only accepts number or numpy ndarray"
    raise nb.TypingError(msg)
