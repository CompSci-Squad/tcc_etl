"""Tests for application settings."""

from __future__ import annotations

import pytest

from tcc_etl.config import Settings, get_settings


class TestSettings:
    def test_defaults(self) -> None:
        s = Settings()
        assert s.aws_region == "us-east-1"
        assert s.s3_bucket_name == "tcc-etl-bucket"
        assert s.batch_size == 10_000
        assert s.max_workers == 4

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        s = Settings()
        assert s.s3_bucket_name == "my-bucket"
        assert s.aws_region == "eu-west-1"

    def test_get_settings_returns_singleton(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
