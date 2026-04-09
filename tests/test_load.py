from __future__ import annotations

import io
from datetime import date, timedelta
from unittest.mock import patch

import boto3
import polars as pl
import pytest
from moto import mock_aws
from pandera.errors import SchemaError

from tcc_etl.loader import (
    FredMdRawSchema,
    FredMdTransformedSchema,
    build_validation_df,
    to_s3,
    validate_and_upload,
    validate_series,
)


def _make_valid_fred_md_df(n: int = 5) -> pl.DataFrame:
    start = date(1990, 1, 1)
    dates = [start + timedelta(days=31 * i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": dates,
            "SERIES1": [float(i) for i in range(n)],
            "SERIES2": [float(i) * 2 for i in range(n)],
        }
    ).with_columns(pl.col("date").cast(pl.Date))


class TestToS3:
    async def test_object_created(self) -> None:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-etl-bucket")
            df = _make_valid_fred_md_df()
            with patch("tcc_etl.loader._s3", client):
                await to_s3(df, "test-etl-bucket", "test/data.parquet")
            obj = client.get_object(Bucket="test-etl-bucket", Key="test/data.parquet")
            assert obj["ContentLength"] > 0

    async def test_content_is_valid_parquet(self) -> None:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-etl-bucket")
            df = _make_valid_fred_md_df()
            with patch("tcc_etl.loader._s3", client):
                await to_s3(df, "test-etl-bucket", "test/data.parquet")
            body = client.get_object(Bucket="test-etl-bucket", Key="test/data.parquet")["Body"].read()
            recovered = pl.read_parquet(io.BytesIO(body))
            assert recovered.schema == df.schema

    async def test_roundtrip_equality(self) -> None:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-etl-bucket")
            df = _make_valid_fred_md_df()
            with patch("tcc_etl.loader._s3", client):
                await to_s3(df, "test-etl-bucket", "test/roundtrip.parquet")
            body = client.get_object(Bucket="test-etl-bucket", Key="test/roundtrip.parquet")["Body"].read()
            recovered = pl.read_parquet(io.BytesIO(body))
            assert df.equals(recovered)


class TestValidateSeries:
    def test_null_rate_correct(self) -> None:
        s = pl.Series("X", [1.0, None, None, None, None])
        rec = validate_series(s)
        assert rec["null_rate"] == pytest.approx(0.8)

    def test_flag_sparse_true_over_50pct(self) -> None:
        s = pl.Series("X", [1.0] + [None] * 9)
        rec = validate_series(s)
        assert rec["flag_sparse"] is True

    def test_flag_sparse_false_under_50pct(self) -> None:
        s = pl.Series("X", [1.0, 2.0, 3.0, None, 5.0])
        rec = validate_series(s)
        assert rec["flag_sparse"] is False

    def test_adf_pvalue_present_for_large_series(self) -> None:
        import numpy as np
        rng = np.random.default_rng(42)
        vals = list(rng.standard_normal(50).cumsum())
        s = pl.Series("Y", vals)
        rec = validate_series(s)
        assert rec["adf_pvalue"] is not None

    def test_no_adf_for_small_series(self) -> None:
        s = pl.Series("Z", [1.0, 2.0, 3.0])
        rec = validate_series(s)
        assert rec["adf_pvalue"] is None
        assert rec["flag_nonstationary"] is False

    def test_series_id_in_output(self) -> None:
        s = pl.Series("MYSERIES", [1.0, 2.0, 3.0])
        rec = validate_series(s)
        assert rec["series_id"] == "MYSERIES"

    def test_expected_keys_present(self) -> None:
        s = pl.Series("A", [1.0, 2.0])
        rec = validate_series(s)
        expected = {
            "series_id", "n_obs", "null_rate", "flag_sparse",
            "flag_infinite", "adf_pvalue", "flag_nonstationary",
        }
        assert set(rec.keys()) == expected


class TestBuildValidationDf:
    def test_row_per_series(self, sample_raw_df: pl.DataFrame, sample_series_ids: list[str]) -> None:
        result = build_validation_df(sample_raw_df, sample_series_ids)
        assert len(result) == len(sample_series_ids)

    def test_columns_correct(self, sample_raw_df: pl.DataFrame, sample_series_ids: list[str]) -> None:
        result = build_validation_df(sample_raw_df, sample_series_ids)
        expected_cols = {
            "series_id", "n_obs", "null_rate", "flag_sparse",
            "flag_infinite", "adf_pvalue", "flag_nonstationary",
        }
        assert set(result.columns) == expected_cols

    def test_skips_missing_series(self, sample_raw_df: pl.DataFrame) -> None:
        result = build_validation_df(sample_raw_df, ["SERIES1", "NONEXISTENT"])
        assert len(result) == 1
        assert result["series_id"][0] == "SERIES1"


class TestFredMdRawSchema:
    def test_passes_valid_df(self) -> None:
        FredMdRawSchema.validate(_make_valid_fred_md_df())

    def test_rejects_non_unique_date(self) -> None:
        df = pl.DataFrame(
            {"date": [date(1990, 1, 1), date(1990, 1, 1)], "SERIES1": [1.0, 2.0]}
        ).with_columns(pl.col("date").cast(pl.Date))
        with pytest.raises(SchemaError):
            FredMdRawSchema.validate(df)

    def test_rejects_null_date(self) -> None:
        df = pl.DataFrame(
            {"date": [None, date(1990, 2, 1)], "SERIES1": [1.0, 2.0]}
        ).with_columns(pl.col("date").cast(pl.Date))
        with pytest.raises(SchemaError):
            FredMdRawSchema.validate(df)


class TestFredMdTransformedSchema:
    def test_passes_valid_df(self) -> None:
        FredMdTransformedSchema.validate(_make_valid_fred_md_df())

    def test_passes_df_with_nulls_in_series(self) -> None:
        df = pl.DataFrame(
            {"date": [date(1990, 1, 1), date(1990, 2, 1)], "SERIES1": [None, 0.01]}
        ).with_columns(pl.col("date").cast(pl.Date))
        FredMdTransformedSchema.validate(df)


class TestValidateAndUpload:
    async def test_uploads_to_s3_on_valid_raw(self) -> None:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-etl-bucket")
            df = _make_valid_fred_md_df()
            with patch("tcc_etl.loader._s3", client):
                await validate_and_upload(df, "test-etl-bucket", "test/raw.parquet", "raw")
            keys = [o["Key"] for o in client.list_objects(Bucket="test-etl-bucket")["Contents"]]
            assert "test/raw.parquet" in keys

    async def test_uploads_to_s3_on_valid_transformed(self) -> None:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-etl-bucket")
            df = _make_valid_fred_md_df()
            with patch("tcc_etl.loader._s3", client):
                await validate_and_upload(df, "test-etl-bucket", "test/trf.parquet", "transformed")
            keys = [o["Key"] for o in client.list_objects(Bucket="test-etl-bucket")["Contents"]]
            assert "test/trf.parquet" in keys

    async def test_raises_schema_error_before_upload_on_invalid(self) -> None:
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-etl-bucket")
            df = pl.DataFrame(
                {"date": [None, date(1990, 2, 1)], "S1": [1.0, 2.0]}
            ).with_columns(pl.col("date").cast(pl.Date))
            with patch("tcc_etl.loader._s3", client):
                with pytest.raises(SchemaError):
                    await validate_and_upload(df, "test-etl-bucket", "test/bad.parquet", "raw")
