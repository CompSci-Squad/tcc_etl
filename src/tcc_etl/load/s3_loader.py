"""S3 loader - uploads data to an AWS S3 bucket.

Supports :class:`polars.DataFrame` (serialised as Parquet or CSV), raw
``bytes``, and arbitrary objects serialised as JSON.
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import boto3
import polars as pl
from loguru import logger

from tcc_etl.config import get_settings
from tcc_etl.load.base import BaseLoader


class S3Loader(BaseLoader):
    """Load data into an S3 bucket.

    Parameters
    ----------
    bucket:
        Target bucket name. Defaults to ``settings.s3_bucket_name``.
    prefix:
        Key prefix applied to every object. Defaults to ``settings.s3_prefix``.
    output_format:
        Serialisation format for :class:`polars.DataFrame` payloads:
        ``"parquet"`` (default, best performance) or ``"csv"``.
    max_workers:
        Concurrency limit for :meth:`load_many`. Defaults to
        ``settings.max_workers``.
    """

    def __init__(
        self,
        *,
        bucket: str | None = None,
        prefix: str | None = None,
        output_format: str = "parquet",
        max_workers: int | None = None,
    ) -> None:
        settings = get_settings()
        self.bucket = bucket or settings.s3_bucket_name
        self.prefix = prefix or settings.s3_prefix
        self.output_format = output_format
        self.max_workers = max_workers or settings.max_workers

        self._s3 = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value() or None,
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value() or None,
            endpoint_url=settings.aws_endpoint_url,
        )

    # ------------------------------------------------------------------
    async def load(self, data: Any, key: str) -> None:  # noqa: ANN401
        """Upload *data* to S3 under ``{prefix}{key}``.

        Parameters
        ----------
        data:
            :class:`polars.DataFrame`, ``bytes``, or a JSON-serialisable object.
        key:
            Object key (without the prefix).
        """
        full_key = f"{self.prefix}{key}"
        body, content_type = self._serialise(data)
        logger.info("Uploading s3://{}/{}", self.bucket, full_key)
        await asyncio.to_thread(
            self._s3.put_object,
            Bucket=self.bucket,
            Key=full_key,
            Body=body,
            ContentType=content_type,
        )
        logger.info("Upload complete - s3://{}/{}", self.bucket, full_key)

    # ------------------------------------------------------------------
    async def load_many(
        self, items: list[tuple[Any, str]], *, max_workers: int | None = None
    ) -> None:
        """Upload multiple objects concurrently.

        Parameters
        ----------
        items:
            List of ``(data, key)`` pairs passed to :meth:`load`.
        max_workers:
            Override the instance-level concurrency limit for this call only.
            Note: asyncio.gather does not enforce a hard cap; pass a
            :class:`asyncio.Semaphore` to limit parallelism if needed.
        """
        workers = max_workers or self.max_workers
        logger.info("Uploading {} objects (concurrency={})", len(items), workers)
        sem = asyncio.Semaphore(workers)

        async def _upload(data: Any, key: str) -> None:  # noqa: ANN401
            async with sem:
                await self.load(data, key)

        results = await asyncio.gather(
            *[_upload(data, key) for data, key in items],
            return_exceptions=True,
        )
        for (_, key), result in zip(items, results, strict=True):
            if isinstance(result, BaseException):
                logger.error("Failed to upload key '{}': {}", key, result)
                raise result

    # ------------------------------------------------------------------
    def _serialise(self, data: Any) -> tuple[bytes, str]:  # noqa: ANN401
        """Return ``(body_bytes, content_type)`` for *data*."""
        if isinstance(data, pl.DataFrame):
            return self._serialise_dataframe(data)
        if isinstance(data, bytes):
            return data, "application/octet-stream"
        # Fallback: JSON-encode anything else
        body = json.dumps(data, default=str).encode()
        return body, "application/json"

    def _serialise_dataframe(self, df: pl.DataFrame) -> tuple[bytes, str]:
        """Serialise a Polars DataFrame to Parquet or CSV bytes."""
        buf = io.BytesIO()
        if self.output_format == "parquet":
            df.write_parquet(buf)
            return buf.getvalue(), "application/octet-stream"
        # CSV fallback
        df.write_csv(buf)
        return buf.getvalue(), "text/csv"
