"""Implements an extended version of the `xlogx` function.

.. autosummary::
   :nosignatures:

   ~xlogx

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from functools import singledispatch
from numbers import Number
from typing import TYPE_CHECKING, Any

import numba as nb
import numpy as np
from numba.extending import overload as numba_overload
from numba.extending import register_jitable
from numpy.typing import NDArray

if TYPE_CHECKING:
    import jax

NoneType = type(None)
FloatingArray = NDArray[np.floating]
FloatOrArray = float | FloatingArray


def _raise_if_negative(x):
    """Helper function to raise exceptions in jax backend."""
    import jax.numpy as jnp

    if jnp.any(x < 0):
        msg = "xlogx expects non-negative values"
        raise ValueError(msg)


@register_jitable
def _xlogx_diff0_scalar(
    x: float, threshold: float = 0, raise_error: bool = False
) -> float:
    """Evaluate xlogx function for one point."""
    if x > threshold:
        return x * np.log(x)  # type: ignore
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
    return 0.5 * (x - threshold) * (x / threshold + 1) + x * log_threshold  # type: ignore


@nb.njit
def _xlogx_diff0_array_numba(
    x: FloatingArray, threshold: float = 0, raise_error: bool = False
) -> FloatingArray:
    """Evaluate xlogx function for a numpy array."""
    y = np.empty_like(x)
    for i in range(x.size):
        y.flat[i] = _xlogx_diff0_scalar(
            x.flat[i], threshold=threshold, raise_error=raise_error
        )
    return y


def _xlogx_diff0_jax(
    x: jax.Array, threshold: float = 0, raise_error: bool = False
) -> jax.Array:
    """Evaluate xlogx function for a jax array."""
    import jax.numpy as jnp

    if threshold == 0:
        if raise_error:
            jax.debug.callback(_raise_if_negative, x)
        return jnp.where(x > 0, x * jnp.log(x), jnp.where(x == 0, 0.0, jnp.nan))

    log_t = jnp.log(threshold)
    linearized = 0.5 * (x - threshold) * (x / threshold + 1) + x * log_t
    return jnp.where(x > threshold, x * jnp.log(x), linearized)


@register_jitable
def _xlogx_diff1_scalar(
    x: float, threshold: float = 0, raise_error: bool = False
) -> float:
    """Evaluate first derivative of xlogx function for one point."""
    if x > threshold:
        return 1 + np.log(x)  # type: ignore
    if threshold == 0:
        # without linearization
        if x == 0:
            return -np.inf
        if raise_error:
            msg = "xlogx expects non-negative values"
            raise ValueError(msg)
        return math.nan
    # with linearization
    return x / threshold + np.log(threshold)  # type: ignore


@nb.njit
def _xlogx_diff1_array_numba(
    x: FloatingArray, threshold: float = 0, raise_error: bool = False
) -> FloatingArray:
    """Evaluate first derivative of xlogx function for a numpy array."""
    y = np.empty_like(x)
    for i in range(x.size):
        y.flat[i] = _xlogx_diff1_scalar(
            x.flat[i], threshold=threshold, raise_error=raise_error
        )
    return y


def _xlogx_diff1_jax(
    x: jax.Array, threshold: float = 0, raise_error: bool = False
) -> jax.Array:
    """Evaluate first derivative of xlogx function for a jax array."""
    import jax.numpy as jnp

    if threshold == 0:
        if raise_error:
            jax.debug.callback(_raise_if_negative, x)
        return jnp.where(x > 0, 1 + jnp.log(x), jnp.where(x == 0, -jnp.inf, jnp.nan))

    linearized = x / threshold + jnp.log(threshold)
    return jnp.where(x > threshold, 1 + jnp.log(x), linearized)


@register_jitable
def _xlogx_diff2_scalar(
    x: float, threshold: float = 0, raise_error: bool = False
) -> float:
    """Evaluate second derivative of xlogx function for one point."""
    if x > threshold:
        return 1 / x
    if threshold == 0:
        return np.inf
    return 1 / threshold


@nb.njit
def _xlogx_diff2_array_numba(
    x: FloatingArray, threshold: float = 0, raise_error: bool = False
) -> FloatingArray:
    """Evaluate second derivative of xlogx function for a numpy array."""
    y = np.empty_like(x)
    for i in range(x.size):
        y.flat[i] = _xlogx_diff2_scalar(
            x.flat[i], threshold=threshold, raise_error=raise_error
        )
    return y


def _xlogx_diff2_jax(
    x: jax.Array, threshold: float = 0, raise_error: bool = False
) -> jax.Array:
    """Evaluate second derivative of xlogx function for a jax array."""
    import jax.numpy as jnp

    if threshold == 0:
        return jnp.where(x > 0, 1 / x, jnp.inf)
    return jnp.where(x > threshold, 1 / x, 1 / threshold)


@singledispatch
def xlogx(
    x,
    threshold: float = 0,
    diff: int | None = None,
    raise_error: bool = False,
) -> Any:
    r"""Calculate :math:`x \log(x)` and its derivatives.

    This function optionally uses a linear approximation for small :math:`x` to
    approximate the function for small negative values, which otherwise would return
    NaN or raise an error (depending on the flag `raise_error`).

    Note that this function has multiple implementations and can thus be used with
    single numbers, numpy arrays, and in compiled contexts. In particular, the function
    supports compilation with :func:`numba.jit` and :func:`jax.jit`.

    Args:
        x (float or :class:`~numpy.ndarray` or :class:`~jax.Array`):
            The value or array to which the function is applied
        threshold (float):
            Threshold below which the function will be linearized to extend the support
            of the function to negative numbers. Setting `threshold` to zero disables
            the linearization, so negative values raise an error or return NaN.
        diff (int):
            Degree of differentiation of the expression. The default value `None`
            corresponds to `diff=0`.
        raise_error (bool):
            If True and `threshold==0`, an error is raised for non-positive values.

    Returns:
        float: The result
    """
    msg = f"`xlogx` does not support {x.__class__}"
    raise TypeError(msg)


@xlogx.register
def _(
    x: Number,
    threshold: float = 0,
    diff: int | None = None,
    raise_error: bool = False,
) -> float:
    if diff == 0 or diff is None:
        return _xlogx_diff0_scalar(x, threshold, raise_error=raise_error)  # type: ignore
    if diff == 1:
        return _xlogx_diff1_scalar(x, threshold, raise_error=raise_error)  # type: ignore
    if diff == 2:
        return _xlogx_diff2_scalar(x, threshold, raise_error=raise_error)  # type: ignore
    msg = f"diff={diff} is not implemented for scalars"
    raise NotImplementedError(msg)


@xlogx.register
def _(
    x: np.ndarray,
    threshold: float = 0,
    diff: int | None = None,
    raise_error: bool = False,
) -> np.ndarray:
    if diff == 0 or diff is None:
        return _xlogx_diff0_array_numba(x, threshold=threshold, raise_error=raise_error)  # type: ignore
    if diff == 1:
        return _xlogx_diff1_array_numba(x, threshold=threshold, raise_error=raise_error)  # type: ignore
    if diff == 2:
        return _xlogx_diff2_array_numba(x, threshold=threshold, raise_error=raise_error)  # type: ignore
    msg = f"diff={diff} is not implemented for arrays"
    raise NotImplementedError(msg)


try:
    import jax
except ImportError:
    ...
else:
    # register jax version of the function if jax is available
    @xlogx.register
    def _(
        x: jax.Array | jax.core.Tracer,
        threshold: float = 0,
        diff: int | None = None,
        raise_error: bool = False,
    ) -> jax.Array:
        if diff == 0 or diff is None:
            return _xlogx_diff0_jax(x, threshold=threshold, raise_error=raise_error)
        if diff == 1:
            return _xlogx_diff1_jax(x, threshold=threshold, raise_error=raise_error)
        if diff == 2:
            return _xlogx_diff2_jax(x, threshold=threshold, raise_error=raise_error)
        msg = f"diff={diff} is not implemented for jax"
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
        impl = [_xlogx_diff0_scalar, _xlogx_diff1_scalar, _xlogx_diff2_scalar][diff_val]

        def xlogx_scalar(x, threshold=0, diff=None, raise_error=False):
            return impl(x, threshold=threshold, raise_error=raise_error)

        return xlogx_scalar

    if isinstance(x, nb.types.Array):
        # return a vectorized implementation
        impl = [
            _xlogx_diff0_array_numba,
            _xlogx_diff1_array_numba,
            _xlogx_diff2_array_numba,
        ][diff_val]

        def xlogx_array(x, threshold=0, diff=None, raise_error=False):
            return impl(x, threshold=threshold, raise_error=raise_error)

        return xlogx_array

    msg = "Only accepts number or numpy ndarray"
    raise nb.TypingError(msg)
