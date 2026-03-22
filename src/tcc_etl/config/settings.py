"""Application configuration using pydantic-settings."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Central configuration - all values can be overridden via environment variables.

    Variable names are upper-cased versions of the field names.
    Example: ``AWS_REGION=us-east-1 tcc-etl``
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AWS / S3 ──────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_access_key_id: SecretStr = Field(
        default=SecretStr(""), description="AWS access key ID"
    )
    aws_secret_access_key: SecretStr = Field(
        default=SecretStr(""), description="AWS secret access key"
    )
    aws_endpoint_url: str | None = Field(
        default=None,
        description="Optional custom endpoint (e.g. LocalStack http://localhost:4566)",
    )
    s3_bucket_name: str = Field(default="tcc-etl-bucket", description="Target S3 bucket name")
    s3_prefix: str = Field(default="data/", description="Key prefix inside the bucket")

    # ── Pipeline ──────────────────────────────────────────────────────────
    batch_size: int = Field(default=10_000, ge=1, description="Records per batch (chunk_size)")
    max_workers: int = Field(default=4, ge=1, description="Concurrency limit for parallel loads")
    source_url: str = Field(default="", description="Data source URL (set via SOURCE_URL env var)")

    # ── Observability ─────────────────────────────────────────────────────
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Log level")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (created once per process).

    Call ``get_settings.cache_clear()`` in tests to reset between runs.
    """
    return Settings()
