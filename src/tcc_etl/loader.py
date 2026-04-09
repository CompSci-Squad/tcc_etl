"""Loader layer -- Pandera validation, async S3 upload, and quality metadata.

Public API:
- validate_and_upload(df, bucket, key, schema)  -> None  (async)
- to_s3(df, bucket, key)                        -> None  (async)
- validate_series(series)                       -> dict
- build_validation_df(df, series_ids)           -> pl.DataFrame

S3 client is initialised at module level so it is reused across warm Lambda
invocations.  Actual network I/O is dispatched to a thread pool via
asyncio.to_thread so the event loop is never blocked by boto3.

Pandera schemas use DataFrameSchema with a regex catch-all column because
FRED-MD has ~130 dynamically-named series that cannot be declared as fixed
fields in a DataFrameModel.  All FRED-MD series IDs are uppercase (VIXCLS,
DGS10, ...) so `[A-Z].*` (anchored to `^[A-Z].*$` by Pandera) matches every
series column while excluding the lowercase `date` column.
"""

from __future__ import annotations

import asyncio
import io
from typing import Literal

import boto3
import pandera.polars as pa
import polars as pl
from statsmodels.tsa.stattools import adfuller

# -- Module-level S3 client (warm-start reuse) ---------------------------------

_s3 = boto3.client("s3")


# -- Pandera schemas -----------------------------------------------------------

FredMdRawSchema = pa.DataFrameSchema(
    columns={
        "date": pa.Column(pl.Date, nullable=False, unique=True),
        "[A-Z].*": pa.Column(pl.Float64, nullable=True, regex=True),
    },
    name="FredMdRawSchema",
    strict=False,
    ordered=False,
)

FredMdTransformedSchema = pa.DataFrameSchema(
    columns={
        "date": pa.Column(pl.Date, nullable=False, unique=True),
        "[A-Z].*": pa.Column(pl.Float64, nullable=True, regex=True),
    },
    name="FredMdTransformedSchema",
    strict=False,
    ordered=False,
)


# -- Async S3 helpers ----------------------------------------------------------


async def to_s3(df: pl.DataFrame, bucket: str, key: str) -> None:
    """Serialise df to Parquet and upload to S3 without blocking the event loop.

    The boto3 put_object call runs in the default thread pool executor via
    asyncio.to_thread so CPU serialisation and network I/O do not block the
    event loop.  Multiple to_s3 calls can be awaited concurrently with
    asyncio.gather for parallel uploads.
    """
    buf = io.BytesIO()
    df.write_parquet(buf)
    body = buf.getvalue()
    await asyncio.to_thread(_s3.put_object, Bucket=bucket, Key=key, Body=body)


async def validate_and_upload(
    df: pl.DataFrame,
    bucket: str,
    key: str,
    schema: Literal["raw", "transformed"],
) -> None:
    """Validate df with Pandera (sync, fast) then upload to S3 (async).

    Raises pandera.errors.SchemaError before any S3 write is attempted.
    """
    schema_obj = FredMdRawSchema if schema == "raw" else FredMdTransformedSchema
    schema_obj.validate(df)
    await to_s3(df, bucket, key)


# -- Quality metadata (synchronous -- CPU-bound, no I/O) ----------------------


def validate_series(series: pl.Series) -> dict:
    """Compute quality metrics for a single FRED-MD series."""
    vals = series.drop_nulls().to_numpy()
    n_total = max(len(series), 1)
    null_rate = round(series.null_count() / n_total, 4)

    rec: dict = {
        "series_id": series.name,
        "n_obs": len(vals),
        "null_rate": null_rate,
        "flag_sparse": null_rate > 0.50,
        "flag_infinite": bool(series.is_infinite().sum() > 0),
        "adf_pvalue": None,
        "flag_nonstationary": False,
    }

    if len(vals) >= 30:
        p_value = float(adfuller(vals, autolag="AIC")[1])
        rec["adf_pvalue"] = round(p_value, 4)
        rec["flag_nonstationary"] = p_value >= 0.05

    return rec


def build_validation_df(df: pl.DataFrame, series_ids: list[str]) -> pl.DataFrame:
    """Build a per-series quality report as a Polars DataFrame."""
    return pl.DataFrame(
        [validate_series(df[sid]) for sid in series_ids if sid in df.columns]
    )
