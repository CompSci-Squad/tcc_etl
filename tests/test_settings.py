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
        # Secrets should be SecretStr instances, not plain strings
        from pydantic import SecretStr

        assert isinstance(s.aws_access_key_id, SecretStr)
        assert isinstance(s.aws_secret_access_key, SecretStr)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        # cache must be cleared so the new env vars are picked up
        get_settings.cache_clear()
        s = get_settings()
        assert s.s3_bucket_name == "my-bucket"
        assert s.aws_region == "eu-west-1"

    def test_get_settings_returns_singleton(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_secrets_not_exposed_in_repr(self) -> None:
        s = Settings()
        # repr / str of SecretStr must NOT expose the actual value
        assert "**" in repr(s.aws_access_key_id)
        assert "**" in repr(s.aws_secret_access_key)
