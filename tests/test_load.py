"""Tests for tcc_etl.loader — upsert_s3, validate_and_upload, and validators."""

from __future__ import annotations

import io
from datetime import date
from unittest.mock import MagicMock, patch

import boto3
import polars as pl
import pytest
from moto import mock_aws

from tcc_etl.loader import (
    upsert_s3,
    validate_and_upload,
    validate_hy_oas_gap,
    validate_monthly_sparsity,
    validate_no_weekends,
)

BUCKET = "test-etl-bucket"
KEY = "panel_raw.parquet"


def _make_s3_client():
    return boto3.client("s3", region_name="us-east-1")


def _make_df(dates: list[date]) -> pl.DataFrame:
    n = len(dates)
    return pl.DataFrame({
        "date": dates,
        "VIXCLS":       [15.0] * n,
        "DGS10":        [4.0] * n,
        "DTB3":         [5.0] * n,
        "BAMLH0A0HYM2": [3.0] * n,
        "DCOILBRENTEU": [80.0] * n,
        "CPIAUCSL":     [None] * n,
        "FEDFUNDS":     [None] * n,
        "INDPRO":       [None] * n,
        "UNRATE":       [None] * n,
        "^GSPC":        [5100.0] * n,
        "DX-Y.NYB":     [104.0] * n,
        "GC=F":         [3000.0] * n,
    }).with_columns(pl.col("date").cast(pl.Date))


# ── upsert_s3 ──────────────────────────────────────────────────────────────


class TestUpsertS3:
    def test_first_run_creates_file(self, s3_mock) -> None:
        bucket, client = s3_mock
        dates = [date(2026, 3, 25), date(2026, 3, 26)]
        df = _make_df(dates)

        with patch("tcc_etl.loader._s3", client):
            total = upsert_s3(df, KEY, bucket)

        assert total == 2
        obj = client.get_object(Bucket=bucket, Key=KEY)
        result = pl.read_parquet(io.BytesIO(obj["Body"].read()))
        assert len(result) == 2

    def test_incremental_merge_deduplicates(self, s3_mock) -> None:
        bucket, client = s3_mock

        # First run
        existing_dates = [date(2026, 3, 23), date(2026, 3, 24), date(2026, 3, 25)]
        existing_df = _make_df(existing_dates)
        with patch("tcc_etl.loader._s3", client):
            upsert_s3(existing_df, KEY, bucket)

        # Second run: overlaps on 2026-03-25, adds 2026-03-26/27
        new_dates = [date(2026, 3, 25), date(2026, 3, 26), date(2026, 3, 27)]
        new_df = _make_df(new_dates)
        with patch("tcc_etl.loader._s3", client):
            total = upsert_s3(new_df, KEY, bucket)

        # Expected: 2026-03-23, 2026-03-24 (kept) + 3 new rows = 5 total
        assert total == 5

    def test_output_is_sorted_by_date(self, s3_mock) -> None:
        bucket, client = s3_mock
        dates = [date(2026, 3, 27), date(2026, 3, 26), date(2026, 3, 25)]
        df = _make_df(dates)
        with patch("tcc_etl.loader._s3", client):
            upsert_s3(df, KEY, bucket)
        obj = client.get_object(Bucket=bucket, Key=KEY)
        result = pl.read_parquet(io.BytesIO(obj["Body"].read()))
        dates_out = result["date"].to_list()
        assert dates_out == sorted(dates_out)


# ── Post-validation helpers ────────────────────────────────────────────────


class TestValidateMonthlySparsity:
    def test_passes_with_sparse_data(self) -> None:
        # 1 non-null out of 31 rows ~ 3.2% → valid
        n = 31
        df = pl.DataFrame({
            "CPIAUCSL": [1.0] + [None] * (n - 1),
            "FEDFUNDS": [1.0] + [None] * (n - 1),
            "INDPRO":   [1.0] + [None] * (n - 1),
            "UNRATE":   [1.0] + [None] * (n - 1),
        })
        validate_monthly_sparsity(df)  # should not raise

    def test_fails_when_fully_populated(self) -> None:
        n = 31
        df = pl.DataFrame({
            "CPIAUCSL": [1.0] * n,
            "FEDFUNDS": [1.0] * n,
            "INDPRO":   [1.0] * n,
            "UNRATE":   [1.0] * n,
        })
        with pytest.raises(AssertionError, match="fill rate"):
            validate_monthly_sparsity(df)


class TestValidateNoWeekends:
    def test_passes_with_business_days_only(self) -> None:
        df = pl.DataFrame({
            "date": [date(2026, 3, 23), date(2026, 3, 24), date(2026, 3, 25)]
        }).with_columns(pl.col("date").cast(pl.Date))
        validate_no_weekends(df)  # should not raise

    def test_fails_with_weekend_date(self) -> None:
        df = pl.DataFrame({
            "date": [date(2026, 3, 21), date(2026, 3, 22)]  # Saturday, Sunday
        }).with_columns(pl.col("date").cast(pl.Date))
        with pytest.raises(AssertionError, match="weekend"):
            validate_no_weekends(df)


class TestValidateHyOasGap:
    def test_passes_with_few_nulls_post_1997(self) -> None:
        import pandas as pd
        dates = list(pd.bdate_range("1997-01-01", periods=100).date)
        df = pl.DataFrame({
            "date": dates,
            "BAMLH0A0HYM2": [3.0] * 100,
        }).with_columns(pl.col("date").cast(pl.Date))
        validate_hy_oas_gap(df)  # should not raise

    def test_fails_with_too_many_nulls_post_1997(self) -> None:
        import pandas as pd
        dates = list(pd.bdate_range("1997-01-01", periods=300).date)
        df = pl.DataFrame({
            "date": dates,
            "BAMLH0A0HYM2": [None] * 300,
        }).with_columns(pl.col("date").cast(pl.Date))
        with pytest.raises(AssertionError, match="unexpected nulls"):
            validate_hy_oas_gap(df)


# ── validate_and_upload ────────────────────────────────────────────────────


class TestValidateAndUpload:
    def test_pandera_error_propagates(self, s3_mock) -> None:
        """A DataFrame violating the schema must raise SchemaError, not upload."""
        import pandera.errors
        bucket, client = s3_mock
        # VIXCLS must be >= 8.0; pass 1.0 to trigger validation error
        bad_df = _make_df([date(2026, 3, 25)]).with_columns(
            pl.lit(1.0).cast(pl.Float64).alias("VIXCLS")
        )
        with patch("tcc_etl.loader._s3", client):
            with pytest.raises(pandera.errors.SchemaError):
                validate_and_upload(bad_df, KEY, bucket, schema="raw")
