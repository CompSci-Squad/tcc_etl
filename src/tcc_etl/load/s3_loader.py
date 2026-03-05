"""S3 loader - uploads data to an AWS S3 bucket.

Supports ``pandas.DataFrame`` (serialised as Parquet or CSV), raw ``bytes``,
and arbitrary objects serialised as JSON.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import boto3
from loguru import logger

from tcc_etl.config import get_settings
from tcc_etl.load.base import BaseLoader

if TYPE_CHECKING:
    import pandas as pd


class S3Loader(BaseLoader):
    """Load data into an S3 bucket.

    Parameters
    ----------
    bucket:
        Target bucket name. Defaults to ``settings.s3_bucket_name``.
    prefix:
        Key prefix applied to every object. Defaults to ``settings.s3_prefix``.
    format:
        Serialisation format for ``pd.DataFrame`` payloads: ``"parquet"``
        (default, best performance) or ``"csv"``.
    max_workers:
        Thread-pool size for parallel multi-part uploads. Defaults to
        ``settings.max_workers``.
    """

    def __init__(
        self,
        *,
        bucket: str | None = None,
        prefix: str | None = None,
        format: str = "parquet",
        max_workers: int | None = None,
    ) -> None:
        settings = get_settings()
        self.bucket = bucket or settings.s3_bucket_name
        self.prefix = prefix or settings.s3_prefix
        self.format = format
        self.max_workers = max_workers or settings.max_workers

        self._s3 = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            endpoint_url=settings.aws_endpoint_url,
        )

    # ------------------------------------------------------------------
    def load(self, data: Any, key: str) -> None:  # noqa: ANN401
        """Upload *data* to S3 under ``{prefix}{key}``.

        Parameters
        ----------
        data:
            ``pd.DataFrame``, ``bytes``, or a JSON-serialisable object.
        key:
            Object key (without the prefix).
        """
        full_key = f"{self.prefix}{key}"
        body, content_type = self._serialise(data, key)
        logger.info("Uploading s3://{}/{}", self.bucket, full_key)
        self._s3.put_object(
            Bucket=self.bucket,
            Key=full_key,
            Body=body,
            ContentType=content_type,
        )
        logger.info("Upload complete - s3://{}/{}", self.bucket, full_key)

    # ------------------------------------------------------------------
    def load_many(self, items: list[tuple[Any, str]], *, max_workers: int | None = None) -> None:
        """Upload multiple objects in parallel using a thread pool.

        Parameters
        ----------
        items:
            List of ``(data, key)`` pairs passed to :meth:`load`.
        max_workers:
            Override the instance-level thread-pool size for this call only.
        """
        workers = max_workers or self.max_workers
        logger.info("Uploading {} objects with {} workers", len(items), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.load, data, key): key for data, key in items}
            for future in as_completed(futures):
                key = futures[future]
                exc = future.exception()
                if exc:
                    logger.error("Failed to upload key '{}': {}", key, exc)
                    raise exc

    # ------------------------------------------------------------------
    def _serialise(self, data: Any, key: str) -> tuple[bytes, str]:  # noqa: ANN401
        """Return ``(body_bytes, content_type)`` for *data*."""
        try:
            import pandas as pd  # local import to keep it optional
        except ImportError:
            pd = None  # type: ignore[assignment]

        if pd is not None and isinstance(data, pd.DataFrame):
            return self._serialise_dataframe(data)

        if isinstance(data, bytes):
            return data, "application/octet-stream"

        # Fallback: JSON-encode anything else
        body = json.dumps(data, default=str).encode()
        return body, "application/json"

    def _serialise_dataframe(self, df: pd.DataFrame) -> tuple[bytes, str]:  # type: ignore[name-defined]
        """Serialise a DataFrame to Parquet or CSV bytes."""
        buf = io.BytesIO()
        if self.format == "parquet":
            df.to_parquet(buf, index=False, engine="pyarrow")
            return buf.getvalue(), "application/octet-stream"
        # CSV fallback
        df.to_csv(buf, index=False)
        return buf.getvalue(), "text/csv"
