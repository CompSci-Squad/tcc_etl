"""Tests for tcc_etl.main.handler — Lambda entrypoint."""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import MagicMock, patch

import polars as pl
import pytest


@pytest.fixture()
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setenv("PANEL_RAW_KEY", "panel_raw.parquet")
    monkeypatch.setenv("PANEL_TRANSFORMED_KEY", "panel_transformed.parquet")


def _make_dummy_df() -> pl.DataFrame:
    return pl.DataFrame({"date": [date(2026, 3, 25)]}).with_columns(
        pl.col("date").cast(pl.Date)
    )


class TestHandler:
    def test_returns_required_keys(self, _env: None) -> None:
        dummy_df = _make_dummy_df()

        with (
            patch("tcc_etl.main.fetch_fred", return_value={}),
            patch("tcc_etl.main.fetch_yahoo", return_value={}),
            patch("tcc_etl.main.build_spine", return_value=dummy_df),
            patch("tcc_etl.main.assemble", return_value=dummy_df),
            patch("tcc_etl.main.transform_panel", return_value=dummy_df),
            patch("tcc_etl.main.validate_and_upload", return_value=100),
        ):
            from tcc_etl.main import handler
            result = handler({}, None)

        assert result["statusCode"] == 200
        assert "rows_in_panel" in result
        assert "rows_added" in result
        assert "panel_raw_key" in result
        assert "panel_transformed_key" in result

    def test_rows_added_equals_len_transformed(self, _env: None) -> None:
        dummy_df = _make_dummy_df()

        with (
            patch("tcc_etl.main.fetch_fred", return_value={}),
            patch("tcc_etl.main.fetch_yahoo", return_value={}),
            patch("tcc_etl.main.build_spine", return_value=dummy_df),
            patch("tcc_etl.main.assemble", return_value=dummy_df),
            patch("tcc_etl.main.transform_panel", return_value=dummy_df),
            patch("tcc_etl.main.validate_and_upload", return_value=50),
        ):
            from tcc_etl.main import handler
            result = handler({}, None)

        assert result["rows_added"] == len(dummy_df)
        assert result["rows_in_panel"] == 50

    def test_extraction_error_propagates(self, _env: None) -> None:
        """Exceptions must not be swallowed — Lambda needs them to mark failure."""
        with patch("tcc_etl.main.fetch_fred", side_effect=RuntimeError("FRED down")):
            from tcc_etl.main import handler
            with pytest.raises(RuntimeError, match="FRED down"):
                handler({}, None)

    def test_uses_default_key_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        monkeypatch.delenv("PANEL_RAW_KEY", raising=False)
        monkeypatch.delenv("PANEL_TRANSFORMED_KEY", raising=False)

        import importlib
        import tcc_etl.main as m
        importlib.reload(m)  # re-evaluate module-level constants with updated env

        dummy_df = _make_dummy_df()
        with (
            patch.object(m, "fetch_fred", return_value={}),
            patch.object(m, "fetch_yahoo", return_value={}),
            patch.object(m, "build_spine", return_value=dummy_df),
            patch.object(m, "assemble", return_value=dummy_df),
            patch.object(m, "transform_panel", return_value=dummy_df),
            patch.object(m, "validate_and_upload", return_value=10),
        ):
            result = m.handler({}, None)

        assert result["panel_raw_key"] == "panel_raw.parquet"
        assert result["panel_transformed_key"] == "panel_transformed.parquet"
