"""Tests for the extract package."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from tcc_etl.extract import FileExtractor, HttpExtractor


class TestFileExtractor:
    def test_extract_json(self, json_file: Path) -> None:
        extractor = FileExtractor(json_file)
        result = extractor.extract()
        assert isinstance(result, list)
        assert len(result) == 4
        assert result[0]["id"] == 1

    def test_extract_csv(self, csv_file: Path) -> None:
        extractor = FileExtractor(csv_file)
        result = extractor.extract()
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["label"] == "a"

    def test_extract_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x00\x01\x02")
        extractor = FileExtractor(p)
        result = extractor.extract()
        assert isinstance(result, bytes)
        assert result == b"\x00\x01\x02"


class TestHttpExtractor:
    def test_extract_json(self) -> None:
        payload = [{"id": 1, "val": 42}]
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(payload).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tcc_etl.extract.http_extractor.urlopen", return_value=mock_response):
            extractor = HttpExtractor("http://example.com/data.json")
            result = extractor.extract()

        assert result == payload

    def test_extract_bytes(self) -> None:
        raw = b"hello bytes"
        mock_response = MagicMock()
        mock_response.read.return_value = raw
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("tcc_etl.extract.http_extractor.urlopen", return_value=mock_response):
            extractor = HttpExtractor("http://example.com/data.bin", as_json=False)
            result = extractor.extract()

        assert result == raw
