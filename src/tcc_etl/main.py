"""AWS Lambda entrypoint for the FRED-MD monthly ETL pipeline.

EventBridge Scheduler trigger: cron(0 22 20 * ? *) UTC  (20th of each month)
Architecture: ARM64 (Graviton), Docker container on AWS Lambda

The internal _handler coroutine is fully async:
- fetch_fred_md() streams the HTTP response line-by-line (httpx)
- Three S3 uploads run concurrently via asyncio.gather (no sequential blocking)

The public handler() wrapper calls asyncio.run() so the Lambda runtime sees a
standard synchronous function.

S3 output layout:
    s3://<BUCKET>/fred_md/
    |-- raw/year=YYYY/month=MM/         fred_md_raw_YYYY_MM.parquet
    |-- transformed/year=YYYY/month=MM/ fred_md_transformed_YYYY_MM.parquet
    +-- metadata/year=YYYY/month=MM/    fred_md_validation_YYYY_MM.parquet
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from tcc_etl.extract import fetch_fred_md
from tcc_etl.loader import build_validation_df, to_s3, validate_and_upload
from tcc_etl.transform import remove_outliers, transform_all

BUCKET: str = os.environ["S3_BUCKET"]


async def _handler(event: dict, context: object) -> dict:
    """Async implementation of the FRED-MD ETL pipeline."""
    lf, tcodes, series_ids = await fetch_fred_md()

    lf_clean = remove_outliers(lf, series_ids)
    df_raw = lf_clean.collect()

    lf_transformed = transform_all(lf_clean, tcodes, series_ids)
    df_transformed = lf_transformed.collect()

    df_validation = build_validation_df(df_transformed, series_ids)

    now = datetime.now(tz=timezone.utc)
    yr = now.strftime("%Y")
    mo = now.strftime("%m")
    tag = f"{yr}_{mo}"
    pfx = f"year={yr}/month={mo}"

    # Upload all three Parquet files concurrently
    await asyncio.gather(
        validate_and_upload(
            df_raw,
            BUCKET,
            f"fred_md/raw/{pfx}/fred_md_raw_{tag}.parquet",
            "raw",
        ),
        validate_and_upload(
            df_transformed,
            BUCKET,
            f"fred_md/transformed/{pfx}/fred_md_transformed_{tag}.parquet",
            "transformed",
        ),
        to_s3(
            df_validation,
            BUCKET,
            f"fred_md/metadata/{pfx}/fred_md_validation_{tag}.parquet",
        ),
    )

    return {
        "statusCode": 200,
        "series": len(series_ids),
        "rows": len(df_transformed),
        "year": yr,
        "month": mo,
    }


def handler(event: dict, context: object) -> dict:
    """AWS Lambda entrypoint -- synchronous wrapper around the async pipeline."""
    return asyncio.run(_handler(event, context))
