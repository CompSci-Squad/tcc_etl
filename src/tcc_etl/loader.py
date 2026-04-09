from __future__ import annotations

import asyncio
import io
from typing import Literal

import boto3
import pandera.polars as pa
import polars as pl
from statsmodels.tsa.stattools import adfuller

_s3 = boto3.client("s3")



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



async def to_s3(df: pl.DataFrame, bucket: str, key: str) -> None:
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
    schema_obj = FredMdRawSchema if schema == "raw" else FredMdTransformedSchema
    schema_obj.validate(df)
    await to_s3(df, bucket, key)



def validate_series(series: pl.Series) -> dict:
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
    return pl.DataFrame(
        [validate_series(df[sid]) for sid in series_ids if sid in df.columns]
    )
