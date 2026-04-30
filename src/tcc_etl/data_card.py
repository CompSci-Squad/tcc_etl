"""Per-series data card.

Builds an audit-friendly summary of every series that survives the
post-tcode panel and is fed into the EM imputer. Used by the thesis to
defend imputation choices and by ``tcc_ai`` to filter training windows.

For each series we record:

- ``series_id``
- ``first_obs_date``: first non-null monthly observation
- ``last_obs_date``: last non-null monthly observation
- ``n_obs``: number of non-null observations
- ``n_missing_leading``: count of NaNs strictly before ``first_obs_date``
- ``n_missing_internal``: count of NaNs at-or-after ``first_obs_date``
- ``frac_missing``: total NaN fraction
- ``frac_missing_leading``
- ``frac_missing_internal``
- ``kept``: True if the series passes ``max_missing_frac``
- ``drop_reason``: non-null only when ``kept`` is False
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def build_data_card(
    panel_lf: pl.LazyFrame,
    series_ids: list[str],
    *,
    kept_series: list[str],
    dropped_series: list[str],
    max_missing_frac: float,
) -> pl.DataFrame:
    """Build the per-series data card from the post-tcode, pre-impute panel."""
    df = panel_lf.collect()
    if "date" not in df.columns:
        raise ValueError("panel_lf must contain a 'date' column")

    dates: list[date] = df["date"].to_list()
    kept_set = set(kept_series)
    rows: list[dict] = []

    for sid in series_ids:
        if sid not in df.columns:
            continue
        s = df[sid]
        vals = s.to_numpy()
        # Boolean per row: is this cell observed?
        is_obs = ~np.isnan(vals) if vals.dtype.kind == "f" else s.is_not_null().to_numpy()
        n_obs = int(is_obs.sum())

        if n_obs == 0:
            first_obs_dt = None
            last_obs_dt = None
            n_missing_leading = int((~is_obs).sum())
            n_missing_internal = 0
        else:
            first_idx = int(np.argmax(is_obs))
            last_idx = int(len(is_obs) - 1 - np.argmax(is_obs[::-1]))
            first_obs_dt = dates[first_idx]
            last_obs_dt = dates[last_idx]
            n_missing_leading = int((~is_obs[:first_idx]).sum())
            n_missing_internal = int((~is_obs[first_idx:]).sum())

        n_total = len(vals)
        n_missing = n_missing_leading + n_missing_internal

        kept = sid in kept_set
        if kept:
            drop_reason = None
        elif sid in dropped_series:
            drop_reason = f"missing_frac>{max_missing_frac}"
        else:
            drop_reason = "not_in_panel"

        rows.append(
            {
                "series_id": sid,
                "first_obs_date": first_obs_dt,
                "last_obs_date": last_obs_dt,
                "n_obs": n_obs,
                "n_missing_leading": n_missing_leading,
                "n_missing_internal": n_missing_internal,
                "frac_missing": round(n_missing / max(n_total, 1), 4),
                "frac_missing_leading": round(n_missing_leading / max(n_total, 1), 4),
                "frac_missing_internal": round(n_missing_internal / max(n_total, 1), 4),
                "kept": kept,
                "drop_reason": drop_reason,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "series_id": pl.String,
            "first_obs_date": pl.Date,
            "last_obs_date": pl.Date,
            "n_obs": pl.Int64,
            "n_missing_leading": pl.Int64,
            "n_missing_internal": pl.Int64,
            "frac_missing": pl.Float64,
            "frac_missing_leading": pl.Float64,
            "frac_missing_internal": pl.Float64,
            "kept": pl.Boolean,
            "drop_reason": pl.String,
        },
    )


def balanced_subpanel_columns(
    data_card: pl.DataFrame,
    *,
    cutoff: date = date(1965, 1, 1),
) -> list[str]:
    """Return series ids whose first observation is on or before ``cutoff``.

    These series have no leading-imputed cells in the post-``cutoff`` portion
    of the panel and form the "balanced" subpanel used as a thesis robustness
    check.
    """
    return (
        data_card.filter(pl.col("kept") & (pl.col("first_obs_date") <= cutoff))
        .get_column("series_id")
        .to_list()
    )
