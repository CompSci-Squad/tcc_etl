from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from tcc_etl.extract import fetch_fred_md
from tcc_etl.loader import build_validation_df, to_s3, validate_and_upload
from tcc_etl.transform import remove_outliers, transform_all

BUCKET: str = os.environ["S3_BUCKET"]


async def _handler(event: dict, context: object) -> dict:
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
    return asyncio.run(_handler(event, context))
