"""Application entry point.

Run with::

    uv run tcc-etl
    # or
    python -m tcc_etl.main
"""

from __future__ import annotations

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


def main() -> None:
    """Entry point for the ``tcc-etl`` CLI command."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger.info("tcc_etl starting up")
    logger.debug("Settings: {}", settings.model_dump())

    pipeline = Pipeline(
        extractor=HttpExtractor(url=settings.source_url, as_json=True),
        transformer=DataFrameTransformer(drop_duplicates=True, dropna=True),
        loader=S3Loader(
            bucket=settings.s3_bucket_name,
            prefix=settings.s3_prefix,
            format="parquet",
        ),
        destination_key="output.parquet",
    )

    pipeline.run()
    logger.info("tcc_etl finished successfully")


if __name__ == "__main__":
    main()
