"""Shared pytest fixtures for the macroeconomic ETL test suite."""

from __future__ import annotations

import os
from collections.abc import Generator
from datetime import date

import boto3
import polars as pl
import pytest
from moto import mock_aws


# ── AWS credential stubs (prevent accidental real calls) ───────────────────

@pytest.fixture(autouse=True)
def _aws_credentials() -> Generator[None, None, None]:
    """Inject fake AWS credentials for every test."""
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    yield


# ── S3 mock ────────────────────────────────────────────────────────────────

@pytest.fixture()
def s3_mock() -> Generator[tuple[str, boto3.client], None, None]:  # type: ignore[type-arg]
    """Start moto S3 mock, create a test bucket, yield (bucket_name, client)."""
    bucket_name = "test-etl-bucket"
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=bucket_name)
        yield bucket_name, client


# ── Business-day date helpers ──────────────────────────────────────────────

# 5 consecutive business days (Mon 2026-03-23 → Fri 2026-03-27)
_BDAYS = [
    date(2026, 3, 23),
    date(2026, 3, 24),
    date(2026, 3, 25),
    date(2026, 3, 26),
    date(2026, 3, 27),
]


# ── Sample DataFrames ──────────────────────────────────────────────────────

@pytest.fixture()
def sample_raw_df() -> pl.DataFrame:
    """5-row panel_raw DataFrame with values inside PanelRawSchema bounds."""
    return pl.DataFrame({
        "date": _BDAYS,
        "VIXCLS":       [15.0, 16.0, 14.5, 17.0, 18.0],
        "DGS10":        [4.0,  4.1,  4.2,  4.0,  3.9],
        "DTB3":         [5.0,  5.1,  5.0,  5.2,  5.1],
        "BAMLH0A0HYM2": [3.0,  3.1,  3.2,  3.0,  2.9],
        "DCOILBRENTEU": [80.0, 81.0, 79.0, 82.0, 83.0],
        # Monthly series — only 1 non-null row each (raw level, not masked)
        "CPIAUCSL":     [310.0, None, None, None, None],
        "FEDFUNDS":     [5.33,  None, None, None, None],
        "INDPRO":       [103.0, None, None, None, None],
        "UNRATE":       [4.1,   None, None, None, None],
        "^GSPC":        [5100.0, 5120.0, 5090.0, 5150.0, 5200.0],
        "DX-Y.NYB":     [104.0,  104.2,  104.1,  104.3,  104.0],
        "GC=F":         [3000.0, 3010.0, 2990.0, 3020.0, 3030.0],
    }).with_columns(pl.col("date").cast(pl.Date))


@pytest.fixture()
def sample_transformed_df() -> pl.DataFrame:
    """5-row panel_transformed DataFrame with values inside transformed bounds."""
    import math

    prices_raw = [5100.0, 5120.0, 5090.0, 5150.0, 5200.0]
    returns = [None] + [
        math.log(prices_raw[i] / prices_raw[i - 1])
        for i in range(1, len(prices_raw))
    ]

    return pl.DataFrame({
        "date":         _BDAYS,
        "VIXCLS":       [None, 1.0, -1.5, 2.5, 1.0],
        "DGS10":        [None, 0.1, 0.1, -0.2, -0.1],
        "DTB3":         [None, 0.1, -0.1, 0.2, -0.1],
        "BAMLH0A0HYM2": [None, 0.1, 0.1, -0.2, -0.1],
        "DCOILBRENTEU": [None, 0.01, -0.02, 0.03, 0.01],
        "CPIAUCSL":     [None, None, None, None, None],
        "FEDFUNDS":     [None, None, None, None, None],
        "INDPRO":       [None, None, None, None, None],
        "UNRATE":       [None, None, None, None, None],
        "^GSPC":        returns,
        "DX-Y.NYB":     [None, 0.002, -0.001, 0.002, -0.003],
        "GC=F":         [None, 0.003, -0.007, 0.010, 0.003],
    }).with_columns(pl.col("date").cast(pl.Date))

