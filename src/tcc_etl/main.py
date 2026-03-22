"""Application entry point.

Run with::

    uv run tcc-etl
    # or
    python -m tcc_etl.main
"""

from __future__ import annotations

import asyncio
import sys

from loguru import logger

from tcc_etl.config import get_settings
from tcc_etl.extract import HttpExtractor
from tcc_etl.load import S3Loader
from tcc_etl.pipeline import Pipeline
from tcc_etl.transform import DataFrameTransformer


def _configure_logging(level: str) -> None:
    """Set up loguru with sensible defaults."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


async def _async_main() -> None:
    """Async body of the ETL pipeline run."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    if not settings.source_url:
        logger.error(
            "SOURCE_URL is not configured. "
            "Set it via the SOURCE_URL environment variable or .env file."
        )
        sys.exit(1)

    logger.info("tcc_etl starting up")
    # Exclude secrets from debug log
    safe_dump = settings.model_dump(exclude={"aws_access_key_id", "aws_secret_access_key"})
    logger.debug("Settings: {}", safe_dump)

    pipeline = Pipeline(
        extractor=HttpExtractor(url=settings.source_url, as_json=True),
        transformer=DataFrameTransformer(drop_duplicates=True, dropna=True),
        loader=S3Loader(
            bucket=settings.s3_bucket_name,
            prefix=settings.s3_prefix,
            output_format="parquet",
        ),
        destination_key="output.parquet",
    )

    batch_count = await pipeline.run()
    logger.info("tcc_etl finished successfully - {} batch(es) uploaded", batch_count)


def main() -> None:
    """Entry point for the ``tcc-etl`` CLI command."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
