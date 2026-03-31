"""Extract layer — fetch macroeconomic data from FRED and Yahoo Finance.

Two public functions:
- fetch_fred(start, end) -> dict[str, pl.DataFrame]
- fetch_yahoo(start, end) -> dict[str, pl.DataFrame]
"""

from __future__ import annotations

import os

import polars as pl
import yfinance as yf
from fredapi import Fred

# ── Constants ──────────────────────────────────────────────────────────────

FRED_SERIES: list[str] = [
    "VIXCLS",
    "DGS10",
    "DTB3",
    "BAMLH0A0HYM2",
    "DCOILBRENTEU",
    "CPIAUCSL",
    "FEDFUNDS",
    "INDPRO",
    "UNRATE",
]

YAHOO_TICKERS: list[str] = ["^GSPC", "GC=F", "DX-Y.NYB"]


# ── Public API ─────────────────────────────────────────────────────────────


def fetch_fred(start: str, end: str) -> dict[str, pl.DataFrame]:
    """Fetch FRED series for the given date range.

    Parameters
    ----------
    start:
        ISO date string, e.g. ``"2026-03-24"``.
    end:
        ISO date string, e.g. ``"2026-03-31"``.

    Returns
    -------
    dict[str, pl.DataFrame]
        Keys are FRED series IDs. Each DataFrame has columns
        ``["date", <series_id>]`` with dtypes ``[pl.Date, pl.Float64]``.

    Raises
    ------
    KeyError
        If ``FRED_API_KEY`` is not set in the environment.
    """
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    result: dict[str, pl.DataFrame] = {}

    for series_id in FRED_SERIES:
        raw = fred.get_series(series_id, observation_start=start, observation_end=end)
        # Build DataFrame from the Series directly — avoids integer column name
        # produced by reset_index() which Polars cannot rename via string keys.
        df = pl.DataFrame({
            "date": raw.index.to_list(),
            series_id: raw.values.tolist(),
        }).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col(series_id).cast(pl.Float64),
        )
        result[series_id] = df

    return result


def fetch_yahoo(start: str, end: str) -> dict[str, pl.DataFrame]:
    """Fetch closing prices for equity/commodity tickers from Yahoo Finance.

    Parameters
    ----------
    start:
        ISO date string, e.g. ``"2026-03-24"``.
    end:
        ISO date string, e.g. ``"2026-03-31"``.

    Returns
    -------
    dict[str, pl.DataFrame]
        Keys are ticker strings. Each DataFrame has columns
        ``["date", <ticker>]`` with dtypes ``[pl.Date, pl.Float64]``.
    """
    raw = yf.download(
        YAHOO_TICKERS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # Strip timezone from DatetimeIndex
    raw.index = raw.index.tz_localize(None)

    close = raw["Close"]
    result: dict[str, pl.DataFrame] = {}

    for ticker in YAHOO_TICKERS:
        series = close[ticker].reset_index()
        series.columns = ["date", ticker]
        df = (
            pl.from_pandas(series)
            .with_columns(
                pl.col("date").cast(pl.Date),
                pl.col(ticker).cast(pl.Float64),
            )
        )
        result[ticker] = df

    return result
