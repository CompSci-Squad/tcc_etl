from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone
from typing import Literal

import blake3
import numpy as np
import boto3
import pandera.polars as pa
import polars as pl
import pyarrow.parquet as pq
from statsmodels.tsa.stattools import adfuller

_s3 = boto3.client("s3")



class FredMdRawModel(pa.DataFrameModel):
    date: pl.Date = pa.Field(nullable=False, unique=True)
    series_cols: pl.Float64 = pa.Field(alias="[A-Z].*", regex=True, nullable=True)

    class Config:
        strict = False


class FredMdTransformedModel(pa.DataFrameModel):
    date: pl.Date = pa.Field(nullable=False, unique=True)
    series_cols: pl.Float64 = pa.Field(alias="[A-Z].*", regex=True, nullable=True)

    class Config:
        strict = False


class FredMdMaskModel(pa.DataFrameModel):
    """Boolean sidecar with the same shape as the transformed panel.

    ``True`` means the cell was originally NaN and was filled by EM-PCA.
    """

    date: pl.Date = pa.Field(nullable=False, unique=True)
    series_cols: pl.Boolean = pa.Field(alias="[A-Z].*", regex=True, nullable=False)

    class Config:
        strict = False


class DataCardModel(pa.DataFrameModel):
    series_id: pl.String = pa.Field(nullable=False, unique=True)
    first_obs_date: pl.Date = pa.Field(nullable=True)
    last_obs_date: pl.Date = pa.Field(nullable=True)
    n_obs: pl.Int64 = pa.Field(nullable=False, ge=0)
    n_missing_leading: pl.Int64 = pa.Field(nullable=False, ge=0)
    n_missing_internal: pl.Int64 = pa.Field(nullable=False, ge=0)
    frac_missing: pl.Float64 = pa.Field(nullable=False, ge=0.0, le=1.0)
    frac_missing_leading: pl.Float64 = pa.Field(nullable=False, ge=0.0, le=1.0)
    frac_missing_internal: pl.Float64 = pa.Field(nullable=False, ge=0.0, le=1.0)
    kept: pl.Boolean = pa.Field(nullable=False)
    drop_reason: pl.String = pa.Field(nullable=True)

    class Config:
        strict = True



async def to_s3(
    lf: pl.LazyFrame,
    bucket: str,
    key: str,
    *,
    extraction_ts: datetime | None = None,
) -> None:
    buf = io.BytesIO()
    writer: pq.ParquetWriter | None = None
    try:
        for batch in lf.collect_batches(chunk_size=100):
            arrow_table = batch.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(buf, arrow_table.schema, compression="zstd")
            writer.write_table(arrow_table)
    finally:
        if writer is not None:
            writer.close()

    parquet_bytes = buf.getvalue()
    content_hash = blake3.blake3(parquet_bytes).hexdigest()
    written_at = datetime.now(tz=timezone.utc).isoformat()
    schema_names = lf.collect_schema().names()

    metadata: dict[str, str] = {
        "blake3": content_hash,
        "written-at": written_at,
        "extraction-ts": (extraction_ts or datetime.now(tz=timezone.utc)).isoformat(),
    }

    if "date" in schema_names:
        bounds = lf.select(
            pl.col("date").min().alias("first"),
            pl.col("date").max().alias("last"),
        ).collect()
        metadata["date-first"] = bounds["first"][0].strftime("%m/%Y")
        metadata["date-last"] = bounds["last"][0].strftime("%m/%Y")

    buf.seek(0)
    await asyncio.to_thread(
        _s3.upload_fileobj, buf, bucket, key, ExtraArgs={"Metadata": metadata}
    )


async def validate_and_upload(
    lf: pl.LazyFrame,
    bucket: str,
    key: str,
    schema: Literal["raw", "transformed", "mask", "data_card"],
    *,
    extraction_ts: datetime | None = None,
) -> None:
    schema_obj = {
        "raw": FredMdRawModel,
        "transformed": FredMdTransformedModel,
        "mask": FredMdMaskModel,
        "data_card": DataCardModel,
    }[schema]
    schema_obj.validate(lf, lazy=True)
    await to_s3(lf, bucket, key, extraction_ts=extraction_ts)



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
        vals = vals[np.isfinite(vals)]
        p_value = float(adfuller(vals, autolag="AIC")[1])
        rec["adf_pvalue"] = round(p_value, 4)
        rec["flag_nonstationary"] = p_value >= 0.05

    return rec


def build_validation_df(df: pl.DataFrame, series_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [validate_series(df[sid]) for sid in series_ids if sid in df.columns]
    )
