from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime, timezone

from tcc_etl.data_card import balanced_subpanel_columns, build_data_card
from tcc_etl.extract import fetch_fred_md
from tcc_etl.imputation import impute_lazyframe
from tcc_etl.loader import build_validation_df, to_s3, validate_and_upload
from tcc_etl.transform import remove_outliers, transform_all

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

BUCKET: str = os.environ["S3_BUCKET"]
_IMPUTE_K: int = int(os.environ.get("IMPUTE_K", "8"))
_IMPUTE_MAX_MISSING_FRAC: float = float(os.environ.get("IMPUTE_MAX_MISSING_FRAC", "0.5"))
_BALANCED_CUTOFF: date = date.fromisoformat(
    os.environ.get("BALANCED_CUTOFF_DATE", "1965-01-01")
)
_PIPELINE_VERSION: str = os.environ.get("PIPELINE_VERSION", "v2")


async def _handler(event: dict, context: object) -> dict:
    lf, tcodes, series_ids = await fetch_fred_md()

    lf_clean = remove_outliers(lf, series_ids)
    lf_transformed_raw = transform_all(lf_clean, tcodes, series_ids)

    lf_panel, lf_mask, impute_report = impute_lazyframe(
        lf_transformed_raw,
        series_ids,
        k=_IMPUTE_K,
        max_missing_frac=_IMPUTE_MAX_MISSING_FRAC,
    )

    df_panel = lf_panel.collect()
    df_mask = lf_mask.collect()

    data_card = build_data_card(
        lf_transformed_raw,
        series_ids,
        kept_series=impute_report.kept_series,
        dropped_series=impute_report.dropped_series,
        max_missing_frac=_IMPUTE_MAX_MISSING_FRAC,
    )

    balanced_cols = balanced_subpanel_columns(data_card, cutoff=_BALANCED_CUTOFF)
    panel_select = ["date"] + [c for c in df_panel.columns if c in balanced_cols]
    mask_select = ["date"] + [c for c in df_mask.columns if c in balanced_cols]
    df_panel_balanced = df_panel.select(panel_select)
    df_mask_balanced = df_mask.select(mask_select)
    has_balanced = len(balanced_cols) > 0

    df_validation = build_validation_df(df_panel, impute_report.kept_series)

    now = datetime.now(tz=timezone.utc)
    yr = now.strftime("%Y")
    mo = now.strftime("%m")
    tag = f"{yr}_{mo}"
    pfx = f"year={yr}/month={mo}"

    import boto3

    _s3 = boto3.client("s3")
    impute_dict = impute_report.to_dict()
    impute_dict["pipeline_version"] = _PIPELINE_VERSION
    impute_dict["balanced_cutoff_date"] = _BALANCED_CUTOFF.isoformat()
    impute_dict["balanced_n_series"] = len(balanced_cols)
    impute_bytes = json.dumps(impute_dict, indent=2, default=str).encode("utf-8")
    await asyncio.to_thread(
        _s3.put_object,
        Bucket=BUCKET,
        Key=f"fred_md/metadata/{pfx}/fred_md_imputation_{tag}.json",
        Body=impute_bytes,
        ContentType="application/json",
    )

    upload_tasks = [
        validate_and_upload(
            lf_clean,
            BUCKET,
            f"fred_md/raw/{pfx}/fred_md_raw_{tag}.parquet",
            "raw",
            extraction_ts=now,
        ),
        validate_and_upload(
            df_panel.lazy(),
            BUCKET,
            f"fred_md/transformed/{pfx}/fred_md_transformed_{tag}.parquet",
            "transformed",
            extraction_ts=now,
        ),
        validate_and_upload(
            df_mask.lazy(),
            BUCKET,
            f"fred_md/transformed/{pfx}/fred_md_mask_{tag}.parquet",
            "mask",
            extraction_ts=now,
        ),
        validate_and_upload(
            data_card.lazy(),
            BUCKET,
            f"fred_md/metadata/{pfx}/fred_md_data_card_{tag}.parquet",
            "data_card",
            extraction_ts=now,
        ),
        to_s3(
            df_validation.lazy(),
            BUCKET,
            f"fred_md/metadata/{pfx}/fred_md_validation_{tag}.parquet",
            extraction_ts=now,
        ),
    ]
    if has_balanced:
        upload_tasks.extend(
            [
                validate_and_upload(
                    df_panel_balanced.lazy(),
                    BUCKET,
                    f"fred_md/transformed/{pfx}/fred_md_transformed_balanced_{tag}.parquet",
                    "transformed",
                    extraction_ts=now,
                ),
                validate_and_upload(
                    df_mask_balanced.lazy(),
                    BUCKET,
                    f"fred_md/transformed/{pfx}/fred_md_mask_balanced_{tag}.parquet",
                    "mask",
                    extraction_ts=now,
                ),
            ]
        )
    await asyncio.gather(*upload_tasks)

    response = {
        "statusCode": 200,
        "series_input": len(series_ids),
        "series_kept": len(impute_report.kept_series),
        "series_dropped": len(impute_report.dropped_series),
        "balanced_n_series": len(balanced_cols),
        "frac_imputed": impute_report.frac_imputed,
        "frac_imputed_leading": impute_report.frac_imputed_leading,
        "frac_imputed_internal": impute_report.frac_imputed_internal,
        "em_iter": impute_report.n_iter,
        "em_converged": impute_report.converged,
        "rows": len(df_panel),
        "year": yr,
        "month": mo,
        "pipeline_version": _PIPELINE_VERSION,
    }
    logger.info("ETL complete: %s", json.dumps(response, default=str))
    return response


def handler(event: dict, context: object) -> dict:
    return asyncio.run(_handler(event, context))
