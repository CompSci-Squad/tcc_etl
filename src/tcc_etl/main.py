"""AWS Lambda entrypoint for the macroeconomic ETL pipeline.

The handler performs an incremental 7-day windowed pull from FRED and
Yahoo Finance, assembles a business-day panel, applies locked
transformations, validates with Pandera, and upserts two Parquet files
(panel_raw, panel_transformed) to S3.

EventBridge Scheduler trigger: cron(0 22 ? * MON-FRI *) UTC
Architecture: ARM64 (Graviton), Docker container on AWS Lambda
"""

from __future__ import annotations

import os
from datetime import date, timedelta

from dotenv import load_dotenv

from tcc_etl.extract import fetch_fred, fetch_yahoo
from tcc_etl.loader import validate_and_upload
from tcc_etl.transform import assemble, build_spine, transform_panel

# ── Module-level config (evaluated once per cold start, reused on warm starts)

load_dotenv()  # no-op in Lambda (env vars set by runtime); useful for local dev

BUCKET: str = os.environ["S3_BUCKET_NAME"]
RAW_KEY: str = os.environ.get("PANEL_RAW_KEY", "panel_raw.parquet")
TRANSFORMED_KEY: str = os.environ.get("PANEL_TRANSFORMED_KEY", "panel_transformed.parquet")


# ── Lambda handler ─────────────────────────────────────────────────────────


def handler(event: dict, context: object) -> dict:  # type: ignore[type-arg]
    """AWS Lambda entrypoint — runs the full incremental ETL pipeline.

    Parameters
    ----------
    event:
        EventBridge event payload (not used; pipeline is triggered on schedule).
    context:
        Lambda context object (not used directly).

    Returns
    -------
    dict
        ``{"statusCode": 200, "rows_in_panel": int, "rows_added": int,
           "panel_raw_key": str, "panel_transformed_key": str}``

    Raises
    ------
    Exception
        Any extraction, transformation, or validation exception propagates
        directly so Lambda marks the invocation as failed and EventBridge
        logs it. Do not catch exceptions here.
    """
    end: str = date.today().isoformat()
    start: str = (date.today() - timedelta(days=7)).isoformat()

    # 1. Extract
    fred_data = fetch_fred(start, end)
    yahoo_data = fetch_yahoo(start, end)

    # 2. Assemble raw panel
    spine = build_spine(start, end)
    panel_raw = assemble(spine, fred_data, yahoo_data)

    # 3. Transform
    panel_transformed = transform_panel(panel_raw)

    # 4. Validate + upsert to S3
    raw_total = validate_and_upload(panel_raw, RAW_KEY, BUCKET, schema="raw")
    trf_total = validate_and_upload(panel_transformed, TRANSFORMED_KEY, BUCKET, schema="transformed")

    return {
        "statusCode": 200,
        "rows_in_panel": trf_total,
        "rows_added": len(panel_transformed),
        "panel_raw_key": RAW_KEY,
        "panel_transformed_key": TRANSFORMED_KEY,
    }

