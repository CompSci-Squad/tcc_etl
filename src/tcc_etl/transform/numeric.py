"""JAX-accelerated numeric transformations.

This module provides JIT-compiled helper functions via **JAX** that operate
on arrays.  JAX compiles these functions to XLA operations on the first call,
yielding accelerator (CPU/GPU/TPU) performance for heavy numeric workloads.

Note
----
When JAX is unavailable the module falls back to pure-Python implementations
so the pipeline can still run in stripped environments.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# JAX JIT compilation with graceful pure-Python fallback.
# ---------------------------------------------------------------------------
try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True

    @jax.jit
    def _normalize_jax(arr: jax.Array) -> jax.Array:
        """Min-max normalize *arr* to [0, 1] via XLA."""
        min_val = jnp.min(arr)
        max_val = jnp.max(arr)
        rng = max_val - min_val
        # Avoid division by zero: when rng==0 every value maps to 0.0
        safe_rng = jnp.where(rng > 0, rng, 1.0)
        return jnp.where(rng > 0, (arr - min_val) / safe_rng, jnp.zeros_like(arr))

    @jax.jit
    def _clip_jax(arr: jax.Array, lo: float, hi: float) -> jax.Array:
        """Clip *arr* to [lo, hi] via XLA."""
        return jnp.clip(arr, min=lo, max=hi)

except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def normalize_column(arr: list[float]) -> list[float]:
    """Return a min-max normalized copy of *arr* (values ∈ [0, 1]).

    Uses the JAX-compiled kernel when available; falls back to pure Python.

    Parameters
    ----------
    arr:
        1-D sequence of floats.

    Returns
    -------
    list[float]
        Normalized values ready to be assigned back to a Polars Series.
    """
    if not arr:
        return arr
    if _JAX_AVAILABLE:
        return _normalize_jax(jnp.array(arr, dtype=jnp.float32)).tolist()  # type: ignore[union-attr]
    # Pure-Python fallback
    min_val = min(arr)
    max_val = max(arr)
    rng = max_val - min_val
    if rng == 0.0:
        return [0.0] * len(arr)
    return [(v - min_val) / rng for v in arr]


def clip_column(arr: list[float], lo: float, hi: float) -> list[float]:
    """Clip *arr* to *[lo, hi]*.

    Uses the JAX-compiled kernel when available; falls back to pure Python.

    Parameters
    ----------
    arr:
        1-D sequence of floats.
    lo:
        Lower bound (inclusive).
    hi:
        Upper bound (inclusive).

    Returns
    -------
    list[float]
        Clipped values ready to be assigned back to a Polars Series.
    """
    if not arr:
        return arr
    if _JAX_AVAILABLE:
        return _clip_jax(jnp.array(arr, dtype=jnp.float32), lo, hi).tolist()  # type: ignore[union-attr]
    # Pure-Python fallback
    return [lo if v < lo else hi if v > hi else v for v in arr]
