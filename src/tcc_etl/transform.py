"""Transform layer — spine construction, panel assembly, and transformations.

Three public functions:
- build_spine(start, end)    -> pl.DataFrame  (business-day date index)
- assemble(spine, fred, yf)  -> pl.DataFrame  (panel_raw, 13 columns)
- transform_panel(panel_raw) -> pl.DataFrame  (panel_transformed)

Transformation rules are locked — do not modify without explicit instruction.
See the project spec for rationale behind each decision.
"""

from __future__ import annotations

import math

from datetime import date

import polars as pl

# ── Column groupings (locked) ──────────────────────────────────────────────

# Rule 1: log-return  →  ln(P_t / P_{t-1})
_LOG_RETURN_COLS: list[str] = ["^GSPC", "GC=F", "DX-Y.NYB", "DCOILBRENTEU"]

# Rule 2: arithmetic first-difference  →  x_t − x_{t-1}
_ARITH_DIFF_COLS: list[str] = ["VIXCLS", "DGS10", "DTB3", "BAMLH0A0HYM2"]

# Rule 3: monthly series — pointwise masking (4-step, see transform_panel)
# FEDFUNDS and UNRATE → arith diff; CPIAUCSL and INDPRO → log-return
# All four are re-masked to publication dates only after transformation.
_MONTHLY_LOG_RETURN: list[str] = ["CPIAUCSL", "INDPRO"]
_MONTHLY_ARITH_DIFF: list[str] = ["FEDFUNDS", "UNRATE"]
_MONTHLY_COLS: list[str] = _MONTHLY_LOG_RETURN + _MONTHLY_ARITH_DIFF

# Canonical column order for the output panel
_PANEL_COLUMNS: list[str] = [
    "date",
    "VIXCLS",
    "DGS10",
    "DTB3",
    "BAMLH0A0HYM2",
    "DCOILBRENTEU",
    "CPIAUCSL",
    "FEDFUNDS",
    "INDPRO",
    "UNRATE",
    "^GSPC",
    "DX-Y.NYB",
    "GC=F",
]


# Batch size for collect_batches in transform_panel
_BATCH_SIZE: int = 5_000

# ── Public API ─────────────────────────────────────────────────────────────


def build_spine(start: str, end: str) -> pl.DataFrame:
    """Return a business-day (Mon–Fri) date spine for the given range.

    Parameters
    ----------
    start, end:
        ISO date strings, e.g. ``"2026-03-24"``.

    Returns
    -------
    pl.DataFrame
        Single column ``date`` with dtype ``pl.Date``, no weekends.
    """
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    return (
        pl.date_range(start=start_date, end=end_date, interval="1d", eager=True)
        .to_frame("date")
        .filter(pl.col("date").dt.weekday() <= 5)
    )


def assemble(
    spine: pl.DataFrame,
    fred_data: dict[str, pl.DataFrame],
    yahoo_data: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    """Left-join all 12 series onto the business-day spine.

    Parameters
    ----------
    spine:
        Output of :func:`build_spine`.
    fred_data:
        Output of :func:`~tcc_etl.extract.fetch_fred`.
    yahoo_data:
        Output of :func:`~tcc_etl.extract.fetch_yahoo`.

    Returns
    -------
    pl.DataFrame
        ``panel_raw`` with 13 columns (date + 12 series) in canonical order.
        All non-date columns are ``pl.Float64``.
    """
    all_series = {**fred_data, **yahoo_data}
    lf = spine.lazy()

    for col in _PANEL_COLUMNS[1:]:  # skip "date"
        if col in all_series:
            lf = lf.join(all_series[col].lazy(), on="date", how="left")

    return lf.select(_PANEL_COLUMNS).collect()


def transform_panel(panel_raw: pl.DataFrame) -> pl.DataFrame:
    """Apply locked transformation rules to produce the model-input panel.

    Rules (must not be reordered):
    1. Log-return for price series.
    2. Arithmetic first-difference for rate/spread series.
    3. Four-step pointwise masking for monthly series
       (masks captured BEFORE forward-fill to avoid all-True masks).

    Parameters
    ----------
    panel_raw:
        Output of :func:`assemble`.

    Returns
    -------
    pl.DataFrame
        ``panel_transformed`` with the same 13 columns.
        Monthly series will be ~96–97% null by design.
    """
    # ── Rule 3 Step 1: capture publication masks BEFORE any forward-fill ──
    # Must use panel_raw (the untouched input). The mask records which dates
    # had observed values.
    pub_masks: dict[str, pl.Series] = {
        c: panel_raw[c].is_not_null()
        for c in _MONTHLY_COLS
    }

    # Build a lazy chain for Rules 1 + 2 + Rule 3 Steps 2 + 3
    lf = panel_raw.lazy()

    # ── Rule 1: log-return ────────────────────────────────────────────────
    lf = lf.with_columns([
        pl.col(c).log(base=math.e).diff().alias(c)
        for c in _LOG_RETURN_COLS
    ])

    # ── Rule 2: arithmetic first-difference ───────────────────────────────
    lf = lf.with_columns([
        pl.col(c).diff().alias(c)
        for c in _ARITH_DIFF_COLS
    ])

    # ── Rule 3 Step 2: forward-fill monthly series ────────────────────────
    lf = lf.with_columns([
        pl.col(c).forward_fill()
        for c in _MONTHLY_COLS
    ])

    # ── Rule 3 Step 3: apply transformations ─────────────────────────────
    lf = lf.with_columns([
        pl.col(c).log(base=math.e).diff().alias(c)
        for c in _MONTHLY_LOG_RETURN
    ])
    lf = lf.with_columns([
        pl.col(c).diff().alias(c)
        for c in _MONTHLY_ARITH_DIFF
    ])

    # Collect via batches
    chunks = list(lf.collect_batches(chunk_size=_BATCH_SIZE))
    panel = pl.concat(chunks) if chunks else lf.collect()

    # ── Rule 3 Step 4: re-mask to publication dates only ─────────────────
    # After this step, monthly columns will be ~96–97% null. This is correct.
    # Do NOT forward-fill after this point — the model handles sparse inputs.
    panel = panel.with_columns([
        pl.when(pub_masks[c]).then(pl.col(c)).otherwise(None).alias(c)
        for c in _MONTHLY_COLS
    ])

    return panel.select(_PANEL_COLUMNS)
