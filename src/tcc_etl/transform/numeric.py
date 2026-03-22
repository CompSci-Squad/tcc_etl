"""Numba-accelerated numeric transformations.

This module provides JIT-compiled helper functions via **Numba** that operate
on NumPy arrays.  Numba compiles these functions to native machine code on
the first call, yielding near-C performance for heavy numeric workloads.

Note
----
Numba is available only on CPython.  When running on PyPy the helpers fall
back to their pure-NumPy counterparts automatically (PyPy already applies
its own JIT compilation).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Numba JIT compilation with graceful fallback for PyPy / environments where
# Numba is not installed.
# ---------------------------------------------------------------------------
try:
    from numba import njit  # type: ignore[import-untyped]

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - PyPy or stripped environment
    _NUMBA_AVAILABLE = False

    def njit(  # type: ignore[misc]
        *_args: object,
        **_kwargs: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Identity decorator - passthrough when numba is unavailable."""

        def decorator(fn: Callable[..., object]) -> Callable[..., object]:  # type: ignore[misc]
            return fn

        return decorator


# ---------------------------------------------------------------------------
# JIT-compiled kernels
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)  # type: ignore[misc]
def _normalize_array(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Min-max normalize *arr* to the [0, 1] range.

    Compiled with ``fastmath=True`` for maximum floating-point throughput.
    Returns *arr* unchanged if all values are identical (zero range).
    """
    min_val = arr[0]
    max_val = arr[0]
    for v in arr:
        if v < min_val:
            min_val = v
        if v > max_val:
            max_val = v
    rng = max_val - min_val
    if rng == 0.0:
        return arr
    result = np.empty_like(arr)
    for i, v in enumerate(arr):
        result[i] = (v - min_val) / rng
    return result


@njit(cache=True, fastmath=True)  # type: ignore[misc]
def _clip_array(
    arr: npt.NDArray[np.float64],
    lo: float,
    hi: float,
) -> npt.NDArray[np.float64]:
    """Clip *arr* values to the [lo, hi] interval (JIT-compiled)."""
    result = np.empty_like(arr)
    for i, v in enumerate(arr):
        if v < lo:
            result[i] = lo
        elif v > hi:
            result[i] = hi
        else:
            result[i] = v
    return result


# ---------------------------------------------------------------------------
# Public helpers (backed by JIT kernels above)
# ---------------------------------------------------------------------------


def normalize_column(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return a min-max normalized copy of *arr* (values ∈ [0, 1]).

    Uses the Numba-compiled kernel when available.
    """
    return _normalize_array(np.asarray(arr, dtype=np.float64))


def clip_column(
    arr: npt.NDArray[np.float64],
    lo: float,
    hi: float,
) -> npt.NDArray[np.float64]:
    """Clip *arr* to *[lo, hi]* using a JIT-compiled loop."""
    return _clip_array(np.asarray(arr, dtype=np.float64), lo, hi)
