from __future__ import annotations

import os

os.environ.setdefault("PANDERA_VALIDATION_DEPTH", "SCHEMA_AND_DATA")
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import polars as pl
import pytest
from moto import mock_aws


@pytest.fixture(autouse=True)
def _aws_credentials() -> Generator[None, None, None]:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("PANDERA_VALIDATION_DEPTH", "SCHEMA_AND_DATA")
    yield


@pytest.fixture()
def s3_mock() -> Generator[tuple[str, boto3.client], None, None]:
    bucket_name = "test-etl-bucket"
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=bucket_name)
        yield bucket_name, client


_SAMPLE_CSV = (
    "sasdate,SERIES1,SERIES2,SERIES3\n"
    "tcode,1,5,2\n"
    "1/1/1990,100.0,200.0,50.0\n"
    "2/1/1990,101.0,202.0,51.0\n"
    "3/1/1990,102.0,205.0,52.0\n"
    "4/1/1990,103.0,207.0,53.0\n"
    "5/1/1990,104.0,210.0,54.0"
)


@pytest.fixture()
def sample_csv_text() -> str:
    return _SAMPLE_CSV


@pytest.fixture()
def sample_tcodes() -> dict[str, int]:
    return {"SERIES1": 1, "SERIES2": 5, "SERIES3": 2}


@pytest.fixture()
def sample_series_ids() -> list[str]:
    return ["SERIES1", "SERIES2", "SERIES3"]


@pytest.fixture()
def sample_raw_df() -> pl.DataFrame:
    from datetime import date
    return pl.DataFrame(
        {
            "date": [
                date(1990, 1, 1),
                date(1990, 2, 1),
                date(1990, 3, 1),
                date(1990, 4, 1),
                date(1990, 5, 1),
            ],
            "SERIES1": [100.0, 101.0, 102.0, 103.0, 104.0],
            "SERIES2": [200.0, 202.0, 205.0, 207.0, 210.0],
            "SERIES3": [50.0, 51.0, 52.0, 53.0, 54.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))


class _FakeStreamResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    async def aiter_bytes(self, chunk_size: int = 65_536) -> AsyncGenerator[bytes, None]:
        yield self._data

    def raise_for_status(self) -> None:
        pass


@pytest.fixture()
def fred_md_stream_mock(sample_csv_text: str) -> Generator[AsyncMock, None, None]:
    fake_stream = _FakeStreamResponse(sample_csv_text.encode("utf-8"))

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=fake_stream)

    mock_instance = AsyncMock()
    mock_instance.__aenter__.return_value = mock_client

    mock_cls = MagicMock(return_value=mock_instance)

    with patch("tcc_etl.extract.httpx.AsyncClient", mock_cls):
        yield mock_client
