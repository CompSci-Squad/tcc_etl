from __future__ import annotations

import asyncio
import io
import json
import os
from datetime import datetime, timezone

from tcc_etl.extract import fetch_fred_md
from tcc_etl.imputation import impute_lazyframe
from tcc_etl.loader import build_validation_df, to_s3, validate_and_upload
from tcc_etl.transform import remove_outliers, transform_all

BUCKET: str = os.environ["S3_BUCKET"]
_IMPUTE_K: int = int(os.environ.get("IMPUTE_K", "8"))
_IMPUTE_MAX_MISSING_FRAC: float = float(os.environ.get("IMPUTE_MAX_MISSING_FRAC", "0.33"))


async def _handler(event: dict, context: object) -> dict:
    lf, tcodes, series_ids = await fetch_fred_md()

    lf_clean = remove_outliers(lf, series_ids)
    lf_transformed_raw = transform_all(lf_clean, tcodes, series_ids)

    lf_transformed, impute_report = impute_lazyframe(
        lf_transformed_raw,
        series_ids,
        k=_IMPUTE_K,
        max_missing_frac=_IMPUTE_MAX_MISSING_FRAC,
    )

    df_transformed = lf_transformed.collect()
    df_validation = build_validation_df(df_transformed, impute_report.kept_series)

    now = datetime.now(tz=timezone.utc)
    yr = now.strftime("%Y")
    mo = now.strftime("%m")
    tag = f"{yr}_{mo}"
    pfx = f"year={yr}/month={mo}"

    import boto3

    _s3 = boto3.client("s3")
    impute_bytes = json.dumps(impute_report.to_dict(), indent=2).encode("utf-8")
    await asyncio.to_thread(
        _s3.put_object,
        Bucket=BUCKET,
        Key=f"fred_md/metadata/{pfx}/fred_md_imputation_{tag}.json",
        Body=impute_bytes,
        ContentType="application/json",
    )

    await asyncio.gather(
        validate_and_upload(
            lf_clean,
            BUCKET,
            f"fred_md/raw/{pfx}/fred_md_raw_{tag}.parquet",
            "raw",
            extraction_ts=now,
        ),
        validate_and_upload(
            lf_transformed,
            BUCKET,
            f"fred_md/transformed/{pfx}/fred_md_transformed_{tag}.parquet",
            "transformed",
            extraction_ts=now,
        ),
        to_s3(
            df_validation.lazy(),
            BUCKET,
            f"fred_md/metadata/{pfx}/fred_md_validation_{tag}.parquet",
            extraction_ts=now,
        ),
    )

    return {
        "statusCode": 200,
        "series_input": len(series_ids),
        "series_kept": len(impute_report.kept_series),
        "series_dropped": len(impute_report.dropped_series),
        "frac_imputed": impute_report.frac_imputed,
        "em_iter": impute_report.n_iter,
        "em_converged": impute_report.converged,
        "rows": len(df_transformed),
        "year": yr,
        "month": mo,
    }


def handler(event: dict, context: object) -> dict:
    return asyncio.run(_handler(event, context))
