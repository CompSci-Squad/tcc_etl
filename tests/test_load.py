"""Tests for the load package (S3Loader via moto)."""

from __future__ import annotations

import json

import boto3
import pandas as pd
from moto import mock_aws

from tcc_etl.load import S3Loader


@mock_aws
class TestS3Loader:
    """All tests run inside a moto mock context."""

    BUCKET = "test-etl-bucket"
    PREFIX = "data/"

    def _make_loader(self, fmt: str = "parquet") -> S3Loader:
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=self.BUCKET)
        return S3Loader(bucket=self.BUCKET, prefix=self.PREFIX, format=fmt)

    def test_load_dataframe_parquet(self) -> None:
        loader = self._make_loader("parquet")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        loader.load(df, "test.parquet")

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=self.BUCKET, Key=f"{self.PREFIX}test.parquet")
        assert obj["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_load_dataframe_csv(self) -> None:
        loader = self._make_loader("csv")
        df = pd.DataFrame({"x": [10, 20]})
        loader.load(df, "test.csv")

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=self.BUCKET, Key=f"{self.PREFIX}test.csv")
        body = obj["Body"].read().decode()
        assert "x" in body

    def test_load_bytes(self) -> None:
        loader = self._make_loader()
        loader.load(b"\x00\x01\x02", "raw.bin")

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=self.BUCKET, Key=f"{self.PREFIX}raw.bin")
        assert obj["Body"].read() == b"\x00\x01\x02"

    def test_load_json_fallback(self) -> None:
        loader = self._make_loader()
        payload = {"key": "value", "num": 42}
        loader.load(payload, "data.json")

        s3 = boto3.client("s3", region_name="us-east-1")
        obj = s3.get_object(Bucket=self.BUCKET, Key=f"{self.PREFIX}data.json")
        body = json.loads(obj["Body"].read())
        assert body == payload

    def test_load_many_parallel(self) -> None:
        loader = self._make_loader()
        items = [(b"chunk-1", "chunk1.bin"), (b"chunk-2", "chunk2.bin")]
        loader.load_many(items, max_workers=2)

        s3 = boto3.client("s3", region_name="us-east-1")
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket=self.BUCKET)["Contents"]]
        assert f"{self.PREFIX}chunk1.bin" in keys
        assert f"{self.PREFIX}chunk2.bin" in keys
