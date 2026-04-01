"""Tests for tcc_etl.transform — build_spine, assemble, transform_panel."""

from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from tcc_etl.transform import (
    _MONTHLY_COLS,
    _PANEL_COLUMNS,
    assemble,
    build_spine,
    transform_panel,
)

START = "2026-03-23"
END = "2026-03-27"


# ── build_spine ────────────────────────────────────────────────────────────


class TestBuildSpine:
    def test_no_weekends_in_output(self) -> None:
        spine = build_spine("2026-03-01", "2026-03-31")
        assert spine.filter(pl.col("date").dt.weekday() > 5).height == 0

    def test_correct_business_day_count(self) -> None:
        # March 23-27 is Mon-Fri = 5 business days
        spine = build_spine(START, END)
        assert len(spine) == 5

    def test_date_dtype_is_pl_date(self) -> None:
        spine = build_spine(START, END)
        assert spine["date"].dtype == pl.Date

    def test_single_column(self) -> None:
        spine = build_spine(START, END)
        assert spine.columns == ["date"]


# ── assemble ───────────────────────────────────────────────────────────────


def _make_series_df(col: str, dates: list[date], values: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, col: values}).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col(col).cast(pl.Float64),
    )


class TestAssemble:
    def test_returns_13_columns(self) -> None:
        spine = build_spine(START, END)
        dates = spine["date"].to_list()
        vals = [1.0] * 5

        fred_data = {c: _make_series_df(c, dates, vals) for c in [
            "VIXCLS", "DGS10", "DTB3", "BAMLH0A0HYM2", "DCOILBRENTEU",
            "CPIAUCSL", "FEDFUNDS", "INDPRO", "UNRATE",
        ]}
        yahoo_data = {c: _make_series_df(c, dates, vals) for c in ["^GSPC", "GC=F", "DX-Y.NYB"]}

        panel = assemble(spine, fred_data, yahoo_data)
        assert panel.columns == _PANEL_COLUMNS

    def test_all_non_date_cols_are_float64(self) -> None:
        spine = build_spine(START, END)
        dates = spine["date"].to_list()
        vals = [1.0] * 5

        fred_data = {c: _make_series_df(c, dates, vals) for c in [
            "VIXCLS", "DGS10", "DTB3", "BAMLH0A0HYM2", "DCOILBRENTEU",
            "CPIAUCSL", "FEDFUNDS", "INDPRO", "UNRATE",
        ]}
        yahoo_data = {c: _make_series_df(c, dates, vals) for c in ["^GSPC", "GC=F", "DX-Y.NYB"]}

        panel = assemble(spine, fred_data, yahoo_data)
        for col in panel.columns:
            if col != "date":
                assert panel[col].dtype == pl.Float64, f"{col} is not Float64"

    def test_row_count_matches_spine(self) -> None:
        spine = build_spine(START, END)
        dates = spine["date"].to_list()
        vals = [1.0] * 5
        fred_data = {c: _make_series_df(c, dates, vals) for c in [
            "VIXCLS", "DGS10", "DTB3", "BAMLH0A0HYM2", "DCOILBRENTEU",
            "CPIAUCSL", "FEDFUNDS", "INDPRO", "UNRATE",
        ]}
        yahoo_data = {c: _make_series_df(c, dates, vals) for c in ["^GSPC", "GC=F", "DX-Y.NYB"]}
        panel = assemble(spine, fred_data, yahoo_data)
        assert len(panel) == len(spine)


# ── transform_panel ────────────────────────────────────────────────────────


def _build_panel_raw(n: int = 30) -> pl.DataFrame:
    """Build a synthetic panel_raw with n business days."""
    from datetime import timedelta
    d = date(2026, 1, 5)  # first Monday of 2026
    bdays: list[date] = []
    while len(bdays) < n:
        if d.weekday() < 5:
            bdays.append(d)
        d += timedelta(days=1)
    dates = bdays
    # Price-like columns (must be >0 for log)
    price_cols = {
        "^GSPC":        [5000.0 + i for i in range(n)],
        "GC=F":         [3000.0 + i for i in range(n)],
        "DX-Y.NYB":     [104.0 + i * 0.01 for i in range(n)],
        "DCOILBRENTEU": [80.0 + i * 0.1 for i in range(n)],
    }
    rate_cols = {
        "VIXCLS":       [15.0 + i * 0.1 for i in range(n)],
        "DGS10":        [4.0 + i * 0.01 for i in range(n)],
        "DTB3":         [5.0 + i * 0.01 for i in range(n)],
        "BAMLH0A0HYM2": [3.0 + i * 0.01 for i in range(n)],
    }
    # Monthly: only 1 non-null per month (~3.3% fill rate for 30 rows)
    def _monthly(n: int) -> list:
        return [100.0 if i % 21 == 0 else None for i in range(n)]

    monthly_cols = {
        "CPIAUCSL": _monthly(n),
        "FEDFUNDS": _monthly(n),
        "INDPRO":   _monthly(n),
        "UNRATE":   _monthly(n),
    }

    return pl.DataFrame({
        "date": dates,
        **price_cols,
        **rate_cols,
        **monthly_cols,
    }).with_columns(pl.col("date").cast(pl.Date)).select(_PANEL_COLUMNS)


class TestTransformPanel:
    def test_rule1_log_return_first_row_is_null(self) -> None:
        panel = _build_panel_raw()
        result = transform_panel(panel)
        assert result["^GSPC"][0] is None

    def test_rule1_log_return_values_correct(self) -> None:
        panel = _build_panel_raw()
        result = transform_panel(panel)
        raw_vals = panel["^GSPC"].to_list()
        expected = math.log(raw_vals[1] / raw_vals[0])
        assert abs(result["^GSPC"][1] - expected) < 1e-10

    def test_rule2_arith_diff_first_row_is_null(self) -> None:
        panel = _build_panel_raw()
        result = transform_panel(panel)
        assert result["VIXCLS"][0] is None

    def test_rule2_arith_diff_values_correct(self) -> None:
        panel = _build_panel_raw()
        result = transform_panel(panel)
        raw_vals = panel["VIXCLS"].to_list()
        expected = raw_vals[1] - raw_vals[0]
        assert abs(result["VIXCLS"][1] - expected) < 1e-10

    def test_rule3_monthly_cols_are_mostly_null(self) -> None:
        """After masking, monthly cols must be ~96-97% null (2-8% filled)."""
        # Use 100 rows for a realistic fill ratio
        panel = _build_panel_raw(n=100)
        result = transform_panel(panel)
        for col in _MONTHLY_COLS:
            n_total = len(result)
            n_filled = result[col].drop_nulls().len()
            pct = n_filled / n_total * 100
            assert pct < 10, f"{col} fill rate {pct:.1f}% is too high; masking may be broken"

    def test_rule3_mask_order_matters_regression(self) -> None:
        """Verify that capturing masks AFTER ffill would produce wrong results.

        If masks were captured after forward-fill, is_not_null() would return
        True for all rows, and the re-masking step would have no effect —
        monthly cols would be fully populated instead of sparse.

        This test simulates the bug: ffill first, then capture mask.
        The result should be all-non-null (bug) vs mostly-null (correct).
        """
        panel = _build_panel_raw(n=100)

        # Simulate the WRONG order: capture mask AFTER ffill
        panel_ffilled = panel.with_columns([
            pl.col(c).forward_fill() for c in _MONTHLY_COLS
        ])
        wrong_masks = {c: panel_ffilled[c].is_not_null() for c in _MONTHLY_COLS}

        # With wrong masks, re-masking restores nothing (all True)
        import math as _math
        panel_wrong = panel_ffilled.with_columns([
            pl.col(c).log(base=_math.e).diff().alias(c)
            for c in ["CPIAUCSL", "INDPRO"]
        ] + [
            pl.col(c).diff().alias(c)
            for c in ["FEDFUNDS", "UNRATE"]
        ])
        panel_wrong = panel_wrong.with_columns([
            pl.when(wrong_masks[c]).then(pl.col(c)).otherwise(None).alias(c)
            for c in _MONTHLY_COLS
        ])

        # The wrong approach fills more rows (the bug)
        wrong_fill_pct = panel_wrong["CPIAUCSL"].drop_nulls().len() / len(panel_wrong) * 100

        # The correct approach fills very few rows
        correct = transform_panel(panel)
        correct_fill_pct = correct["CPIAUCSL"].drop_nulls().len() / len(correct) * 100

        # Wrong approach should have higher fill rate — proves order matters
        assert wrong_fill_pct > correct_fill_pct, (
            "Mask order regression: wrong and correct approaches produced same fill rate. "
            "The mask-before-ffill rule may be broken."
        )

    def test_output_has_same_columns_as_input(self) -> None:
        panel = _build_panel_raw()
        result = transform_panel(panel)
        assert result.columns == _PANEL_COLUMNS
