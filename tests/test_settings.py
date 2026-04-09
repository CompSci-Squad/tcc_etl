"""Tests for environment variable configuration used by the Lambda handler."""

from __future__ import annotations

import importlib
import os

import pytest


class TestEnvConfig:
    def test_missing_s3_bucket_raises_key_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing and reloading main without S3_BUCKET must raise KeyError."""
        monkeypatch.delenv("S3_BUCKET", raising=False)
        import tcc_etl.main as main_mod

        with pytest.raises(KeyError):
            importlib.reload(main_mod)

    def test_s3_bucket_set_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When S3_BUCKET is set, main.BUCKET reflects the value."""
        monkeypatch.setenv("S3_BUCKET", "my-test-bucket")
        import tcc_etl.main as main_mod

        importlib.reload(main_mod)
        assert main_mod.BUCKET == "my-test-bucket"
