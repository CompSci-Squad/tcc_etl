from __future__ import annotations

import numpy as np
import polars as pl

try:
    import jax.numpy as jnp
    from jax import config as _jax_config

    _jax_config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False



def remove_outliers(
    lf: pl.LazyFrame,
    series_ids: list[str],
    k: float = 10.0,
) -> pl.LazyFrame:
    exprs = [
        pl.when(
            (pl.col(c) >= pl.col(c).median() - k * (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)))
            & (pl.col(c) <= pl.col(c).median() + k * (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)))
        )
        .then(pl.col(c))
        .otherwise(None)
        .alias(c)
        for c in series_ids
    ]
    return lf.with_columns(exprs)


def apply_tcode(series: pl.Series, tcode: int) -> pl.Series:
    arr_np = series.to_numpy(allow_copy=True).astype(float)

    if _JAX_AVAILABLE:
        result = _apply_tcode_jax(arr_np, tcode, series.name)
    else:  
        result = _apply_tcode_numpy(arr_np, tcode, series.name)

    return pl.Series(name=series.name, values=np.asarray(result), dtype=pl.Float64)


def transform_all(
    lf: pl.LazyFrame,
    tcodes: dict[str, int],
    series_ids: list[str],
) -> pl.LazyFrame:
    df = lf.collect()

    transformed_cols = [
        apply_tcode(df[sid], tcodes[sid]).alias(sid)
        for sid in series_ids
        if sid in df.columns and sid in tcodes
    ]

    if transformed_cols:
        df = df.with_columns(transformed_cols)

    return df.slice(2).lazy()



def _apply_tcode_jax(arr_np: np.ndarray, tcode: int, name: str) -> np.ndarray:
    nan = jnp.nan
    arr = jnp.array(arr_np)

    if tcode == 1:
        return np.asarray(arr)
    elif tcode == 2:
        return np.asarray(jnp.concatenate([jnp.array([nan]), jnp.diff(arr)]))
    elif tcode == 3:
        d1 = jnp.concatenate([jnp.array([nan]), jnp.diff(arr)])
        return np.asarray(jnp.concatenate([jnp.array([nan]), jnp.diff(d1)]))
    elif tcode == 4:
        return np.asarray(jnp.where(arr > 0, jnp.log(arr), nan))
    elif tcode == 5:
        lg = jnp.where(arr > 0, jnp.log(arr), nan)
        return np.asarray(jnp.concatenate([jnp.array([nan]), jnp.diff(lg)]))
    elif tcode == 6:
        lg = jnp.where(arr > 0, jnp.log(arr), nan)
        d1 = jnp.concatenate([jnp.array([nan]), jnp.diff(lg)])
        return np.asarray(jnp.concatenate([jnp.array([nan]), jnp.diff(d1)]))
    elif tcode == 7:
        rolled = jnp.roll(arr, 1)
        pct = jnp.where(rolled != 0, arr / rolled - 1.0, nan)
        pct = pct.at[0].set(nan)
        return np.asarray(jnp.concatenate([jnp.array([nan]), jnp.diff(pct)]))
    else:
        raise ValueError(f"tcode desconhecido: {tcode} (serie: {name})")



def _apply_tcode_numpy(arr: np.ndarray, tcode: int, name: str) -> np.ndarray:
    nan = np.nan

    if tcode == 1:
        return arr
    elif tcode == 2:
        return np.concatenate([[nan], np.diff(arr)])
    elif tcode == 3:
        d1 = np.concatenate([[nan], np.diff(arr)])
        return np.concatenate([[nan], np.diff(d1)])
    elif tcode == 4:
        return np.where(arr > 0, np.log(arr), nan)
    elif tcode == 5:
        lg = np.where(arr > 0, np.log(arr), nan)
        return np.concatenate([[nan], np.diff(lg)])
    elif tcode == 6:
        lg = np.where(arr > 0, np.log(arr), nan)
        d1 = np.concatenate([[nan], np.diff(lg)])
        return np.concatenate([[nan], np.diff(d1)])
    elif tcode == 7:
        rolled = np.roll(arr, 1)
        pct = np.where(rolled != 0, arr / rolled - 1.0, nan)
        pct[0] = nan
        return np.concatenate([[nan], np.diff(pct)])
    else:
        raise ValueError(f"tcode desconhecido: {tcode} (serie: {name})")
