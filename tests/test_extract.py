"""Tests for the extract package."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tcc_etl.extract import FileExtractor, HttpExtractor


class TestFileExtractor:
    async def test_extract_json(self, json_file: Path) -> None:
        extractor = FileExtractor(json_file)
        result = await extractor.extract()
        assert isinstance(result, list)
        assert len(result) == 4
        assert result[0]["id"] == 1

    async def test_extract_csv(self, csv_file: Path) -> None:
        extractor = FileExtractor(csv_file)
        result = await extractor.extract()
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["label"] == "a"

    async def test_extract_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x00\x01\x02")
        extractor = FileExtractor(p)
        result = await extractor.extract()
        assert isinstance(result, bytes)
        assert result == b"\x00\x01\x02"


class TestHttpExtractor:
    def _make_mock_client(self, *, json_data=None, content=None):
        """Return a patched httpx.AsyncClient context manager."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        if json_data is not None:
            mock_response.json.return_value = json_data
        if content is not None:
            mock_response.content = content

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        return mock_client, mock_response

    def test_invalid_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            HttpExtractor("ftp://example.com/data.json")

    def test_file_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            HttpExtractor("file:///etc/passwd")

    async def test_extract_json(self) -> None:
        payload = [{"id": 1, "val": 42}]
        mock_client, _ = self._make_mock_client(json_data=payload)

        with patch("tcc_etl.extract.http_extractor.httpx.AsyncClient", return_value=mock_client):
            extractor = HttpExtractor("http://example.com/data.json")
            result = await extractor.extract()

        assert result == payload

    async def test_extract_bytes(self) -> None:
        raw = b"hello bytes"
        mock_client, _ = self._make_mock_client(content=raw)

        with patch("tcc_etl.extract.http_extractor.httpx.AsyncClient", return_value=mock_client):
            extractor = HttpExtractor("http://example.com/data.bin", as_json=False)
            result = await extractor.extract()

        assert result == raw

    async def test_http_error_propagates(self) -> None:
        import httpx

        # raise_for_status is called synchronously, so use MagicMock for the response
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.get.return_value = mock_response

        with patch("tcc_etl.extract.http_extractor.httpx.AsyncClient", return_value=mock_client):
            extractor = HttpExtractor("https://example.com/missing")
            with pytest.raises(httpx.HTTPStatusError):
                await extractor.extract()
