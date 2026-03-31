"""Tests for environment variable configuration used by the Lambda handler."""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest


class TestEnvConfig:
    def test_missing_s3_bucket_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
        monkeypatch.setenv("FRED_API_KEY", "test-key")
        import tcc_etl.main as m
        with pytest.raises(KeyError):
            importlib.reload(m)

    def test_panel_raw_key_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S3_BUCKET_NAME", "bucket")
        monkeypatch.setenv("FRED_API_KEY", "key")
        monkeypatch.delenv("PANEL_RAW_KEY", raising=False)
        import tcc_etl.main as m
        importlib.reload(m)
        assert m.RAW_KEY == "panel_raw.parquet"

    def test_panel_transformed_key_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S3_BUCKET_NAME", "bucket")
        monkeypatch.setenv("FRED_API_KEY", "key")
        monkeypatch.delenv("PANEL_TRANSFORMED_KEY", raising=False)
        import tcc_etl.main as m
        importlib.reload(m)
        assert m.TRANSFORMED_KEY == "panel_transformed.parquet"

    def test_missing_fred_api_key_raises_on_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("S3_BUCKET_NAME", "bucket")
        env = {k: v for k, v in os.environ.items() if k != "FRED_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from tcc_etl.extract import fetch_fred
            with pytest.raises(KeyError):
                fetch_fred("2026-03-24", "2026-03-31")
