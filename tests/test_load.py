"""Tests for the load package (S3Loader via moto)."""

from __future__ import annotations

import json
from collections.abc import Generator

import boto3
import polars as pl
import pytest
from moto import mock_aws

from tcc_etl.load import S3Loader

BUCKET = "test-etl-bucket"
PREFIX = "data/"


@pytest.fixture()
def _s3() -> Generator[None, None, None]:
    """Start a moto S3 mock and create the test bucket."""
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield


def _make_loader(fmt: str = "parquet") -> S3Loader:
    return S3Loader(bucket=BUCKET, prefix=PREFIX, output_format=fmt)


class TestS3Loader:
    async def test_load_dataframe_parquet(self, _s3) -> None:
        loader = _make_loader("parquet")
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        await loader.load(df, "test.parquet")

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}test.parquet")
        assert obj["ResponseMetadata"]["HTTPStatusCode"] == 200

    async def test_load_dataframe_csv(self, _s3) -> None:
        loader = _make_loader("csv")
        df = pl.DataFrame({"x": [10, 20]})
        await loader.load(df, "test.csv")

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}test.csv")
        body = obj["Body"].read().decode()
        assert "x" in body

    async def test_load_bytes(self, _s3) -> None:
        loader = _make_loader()
        await loader.load(b"\x00\x01\x02", "raw.bin")

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}raw.bin")
        assert obj["Body"].read() == b"\x00\x01\x02"

    async def test_load_json_fallback(self, _s3) -> None:
        loader = _make_loader()
        payload = {"key": "value", "num": 42}
        await loader.load(payload, "data.json")

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}data.json")
        body = json.loads(obj["Body"].read())
        assert body == payload

    async def test_load_many_parallel(self, _s3) -> None:
        loader = _make_loader()
        items = [(b"chunk-1", "chunk1.bin"), (b"chunk-2", "chunk2.bin")]
        await loader.load_many(items, max_workers=2)

        s3 = boto3.client("s3", region_name="us-east-1")
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]]
        assert f"{PREFIX}chunk1.bin" in keys
        assert f"{PREFIX}chunk2.bin" in keys

    async def test_load_many_error_propagates(self, _s3) -> None:
        """An upload failure in load_many must propagate immediately."""
        from unittest.mock import patch
        import botocore.exceptions

        loader = _make_loader()
        error = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "PutObject"
        )
        with patch.object(loader._s3, "put_object", side_effect=error):
            with pytest.raises(botocore.exceptions.ClientError):
                await loader.load_many([(b"data", "file.bin")])
