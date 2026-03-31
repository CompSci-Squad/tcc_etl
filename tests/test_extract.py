"""Tests for tcc_etl.extract — fetch_fred and fetch_yahoo."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from tcc_etl.extract import FRED_SERIES, YAHOO_TICKERS, fetch_fred, fetch_yahoo

START = "2026-03-24"
END = "2026-03-31"


def _make_fred_mock() -> MagicMock:
    """Return a Fred mock whose get_series returns a 3-row pandas Series."""
    mock_fred = MagicMock()
    idx = pd.to_datetime(["2026-03-25", "2026-03-26", "2026-03-27"])
    mock_fred.get_series.return_value = pd.Series([1.0, 2.0, 3.0], index=idx)
    return mock_fred


def _make_yf_download() -> pd.DataFrame:
    """Return a mock yf.download result matching the expected shape."""
    import numpy as np

    idx = pd.to_datetime(["2026-03-25", "2026-03-26", "2026-03-27"])
    idx = idx.tz_localize("UTC")
    cols = pd.MultiIndex.from_product(
        [["Close"], YAHOO_TICKERS], names=["Price", "Ticker"]
    )
    data = np.random.default_rng(0).uniform(100, 200, (3, len(YAHOO_TICKERS)))
    return pd.DataFrame(data, index=idx, columns=cols)


# ── fetch_fred ─────────────────────────────────────────────────────────────


class TestFetchFred:
    def test_returns_all_series_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        with patch("tcc_etl.extract.Fred", return_value=_make_fred_mock()):
            result = fetch_fred(START, END)
        assert set(result.keys()) == set(FRED_SERIES)

    def test_each_df_has_correct_columns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        with patch("tcc_etl.extract.Fred", return_value=_make_fred_mock()):
            result = fetch_fred(START, END)
        for series_id, df in result.items():
            assert "date" in df.columns
            assert series_id in df.columns

    def test_date_dtype_is_pl_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        with patch("tcc_etl.extract.Fred", return_value=_make_fred_mock()):
            result = fetch_fred(START, END)
        for df in result.values():
            assert df["date"].dtype == pl.Date

    def test_series_dtype_is_float64(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        with patch("tcc_etl.extract.Fred", return_value=_make_fred_mock()):
            result = fetch_fred(START, END)
        for series_id, df in result.items():
            assert df[series_id].dtype == pl.Float64

    def test_missing_api_key_raises_key_error(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "FRED_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(KeyError):
                fetch_fred(START, END)


# ── fetch_yahoo ────────────────────────────────────────────────────────────


class TestFetchYahoo:
    def test_returns_all_ticker_keys(self) -> None:
        with patch("tcc_etl.extract.yf.download", return_value=_make_yf_download()):
            result = fetch_yahoo(START, END)
        assert set(result.keys()) == set(YAHOO_TICKERS)

    def test_each_df_has_date_and_ticker_columns(self) -> None:
        with patch("tcc_etl.extract.yf.download", return_value=_make_yf_download()):
            result = fetch_yahoo(START, END)
        for ticker, df in result.items():
            assert "date" in df.columns
            assert ticker in df.columns

    def test_date_dtype_is_pl_date(self) -> None:
        with patch("tcc_etl.extract.yf.download", return_value=_make_yf_download()):
            result = fetch_yahoo(START, END)
        for df in result.values():
            assert df["date"].dtype == pl.Date

    def test_ticker_dtype_is_float64(self) -> None:
        with patch("tcc_etl.extract.yf.download", return_value=_make_yf_download()):
            result = fetch_yahoo(START, END)
        for ticker, df in result.items():
            assert df[ticker].dtype == pl.Float64
