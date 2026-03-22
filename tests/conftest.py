"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from pathlib import Path

import boto3
import polars as pl
import pytest
from moto import mock_aws


# ---------------------------------------------------------------------------
# Settings cache isolation - must run before every test
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Generator[None, None, None]:
    """Reset the lru_cache on get_settings() before and after each test."""
    from tcc_etl.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Environment stubs - prevent accidental real AWS calls
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _aws_credentials() -> Generator[None, None, None]:
    """Inject fake AWS credentials and a default SOURCE_URL for every test."""
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("SOURCE_URL", "https://example.com/data.json")
    yield


# ---------------------------------------------------------------------------
# S3 mock
# ---------------------------------------------------------------------------
@pytest.fixture()
def s3_bucket() -> Generator[str, None, None]:
    """Start a moto S3 mock and yield the bucket name."""
    bucket_name = "test-etl-bucket"
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=bucket_name)
        yield bucket_name


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_records() -> list[dict]:
    return [
        {"id": 1, "value": 10.0, "label": "a"},
        {"id": 2, "value": 20.0, "label": "b"},
        {"id": 3, "value": 30.0, "label": "c"},
        {"id": 3, "value": 30.0, "label": "c"},  # duplicate
    ]


@pytest.fixture()
def sample_dataframe(sample_records: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(sample_records)


@pytest.fixture()
def json_file(tmp_path: Path, sample_records: list[dict]) -> Path:
    p = tmp_path / "data.json"
    p.write_text(json.dumps(sample_records))
    return p


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    p.write_text("id,value,label\n1,10.0,a\n2,20.0,b\n3,30.0,c\n")
    return p
