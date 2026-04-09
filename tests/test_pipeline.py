"""Integration tests for the FRED-MD ETL pipeline.

Tests target _handler (the async coroutine) directly to avoid nesting
asyncio.run() inside pytest-asyncio's own event loop.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import pytest
from moto import mock_aws

_SAMPLE_CSV = (
    "sasdate,SERIES1,SERIES2,SERIES3\n"
    "tcode,1,5,2\n"
    "1/1/1990,100.0,200.0,50.0\n"
    "2/1/1990,101.0,202.0,51.0\n"
    "3/1/1990,102.0,205.0,52.0\n"
    "4/1/1990,103.0,207.0,53.0\n"
    "5/1/1990,104.0,210.0,54.0"
)


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, *a: object) -> None:
        pass

    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        for line in self._lines:
            yield line

    def raise_for_status(self) -> None:
        pass


def _make_httpx_mock(csv_text: str) -> MagicMock:
    lines = csv_text.splitlines()
    fake_stream = _FakeStreamResponse(lines)
    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=fake_stream)
    mock_instance = AsyncMock()
    mock_instance.__aenter__.return_value = mock_client
    return MagicMock(return_value=mock_instance)


class TestHandler:
    async def test_returns_status_200(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bucket")

            import tcc_etl.main as main_mod
            importlib.reload(main_mod)

            with (
                patch("tcc_etl.extract.httpx.AsyncClient", _make_httpx_mock(_SAMPLE_CSV)),
                patch("tcc_etl.loader._s3", client),
            ):
                result = await main_mod._handler({}, None)

        assert result["statusCode"] == 200

    async def test_writes_three_s3_objects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bucket")

            import tcc_etl.main as main_mod
            importlib.reload(main_mod)

            with (
                patch("tcc_etl.extract.httpx.AsyncClient", _make_httpx_mock(_SAMPLE_CSV)),
                patch("tcc_etl.loader._s3", client),
            ):
                await main_mod._handler({}, None)

            listed = client.list_objects(Bucket="test-bucket")
            keys = [o["Key"] for o in listed["Contents"]]
            assert any("raw" in k for k in keys)
            assert any("transformed" in k for k in keys)
            assert any("metadata" in k for k in keys)

    def test_missing_s3_bucket_raises_key_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("S3_BUCKET", raising=False)
        import tcc_etl.main as main_mod
        with pytest.raises(KeyError):
            importlib.reload(main_mod)

    async def test_response_contains_series_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S3_BUCKET", "test-bucket")
        with mock_aws():
            client = boto3.client("s3", region_name="us-east-1")
            client.create_bucket(Bucket="test-bucket")

            import tcc_etl.main as main_mod
            importlib.reload(main_mod)

            with (
                patch("tcc_etl.extract.httpx.AsyncClient", _make_httpx_mock(_SAMPLE_CSV)),
                patch("tcc_etl.loader._s3", client),
            ):
                result = await main_mod._handler({}, None)

        assert result["series"] == 3
