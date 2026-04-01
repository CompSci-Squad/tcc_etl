"""Loader layer — Pandera validation, S3 upsert, and upload orchestration.

Public API:
- upsert_s3(new_df, key, bucket)                   -> int  (total row count)
- validate_and_upload(df, key, bucket, schema)      -> int  (total row count)

S3 client is initialised at module level so it is reused across warm Lambda
invocations (connection pool is preserved, no repeated TLS handshakes).
"""

from __future__ import annotations

import io
from typing import Literal

import boto3
import pandera.polars as pa
import polars as pl
from pandera.typing.polars import Series

# ── Module-level S3 client (warm-start reuse) ──────────────────────────────

_s3 = boto3.client("s3")


# ── Pandera schemas ────────────────────────────────────────────────────────


class PanelRawSchema(pa.DataFrameModel):
    """Validate raw price/level data before writing to S3."""

    date: Series[pl.Date] = pa.Field(unique=True, nullable=False)
    VIXCLS: Series[pl.Float64] = pa.Field(nullable=True, ge=8.0, le=90.0)
    DGS10: Series[pl.Float64] = pa.Field(nullable=True, ge=-2.0)
    DTB3: Series[pl.Float64] = pa.Field(nullable=True, ge=-2.0)
    BAMLH0A0HYM2: Series[pl.Float64] = pa.Field(nullable=True, ge=0.0, le=30.0)
    DCOILBRENTEU: Series[pl.Float64] = pa.Field(nullable=True, gt=0.0)
    CPIAUCSL: Series[pl.Float64] = pa.Field(nullable=True)
    FEDFUNDS: Series[pl.Float64] = pa.Field(nullable=True, ge=0.0)
    INDPRO: Series[pl.Float64] = pa.Field(nullable=True)
    UNRATE: Series[pl.Float64] = pa.Field(nullable=True, ge=0.0, le=25.0)
    gspc: Series[pl.Float64] = pa.Field(nullable=True, gt=0.0, alias="^GSPC")
    dxy: Series[pl.Float64] = pa.Field(nullable=True, alias="DX-Y.NYB")
    gold: Series[pl.Float64] = pa.Field(nullable=True, gt=0.0, alias="GC=F")

    class Config:
        name = "PanelRawSchema"
        strict = True
        ordered = False


class PanelTransformedSchema(pa.DataFrameModel):
    """Validate transformed (return/diff) data before writing to S3."""

    date: Series[pl.Date] = pa.Field(unique=True, nullable=False)
    gspc: Series[pl.Float64] = pa.Field(nullable=True, ge=-0.25, le=0.15, alias="^GSPC")
    gold: Series[pl.Float64] = pa.Field(nullable=True, ge=-0.20, le=0.20, alias="GC=F")
    dxy: Series[pl.Float64] = pa.Field(nullable=True, ge=-0.05, le=0.05, alias="DX-Y.NYB")
    DCOILBRENTEU: Series[pl.Float64] = pa.Field(nullable=True, ge=-0.70, le=0.50)
    VIXCLS: Series[pl.Float64] = pa.Field(nullable=True, ge=-30.0, le=30.0)
    DGS10: Series[pl.Float64] = pa.Field(nullable=True, ge=-2.0, le=2.0)
    DTB3: Series[pl.Float64] = pa.Field(nullable=True, ge=-2.0, le=2.0)
    BAMLH0A0HYM2: Series[pl.Float64] = pa.Field(nullable=True, ge=-5.0, le=5.0)
    FEDFUNDS: Series[pl.Float64] = pa.Field(nullable=True, ge=-2.0, le=2.0)
    CPIAUCSL: Series[pl.Float64] = pa.Field(nullable=True)
    INDPRO: Series[pl.Float64] = pa.Field(nullable=True)
    UNRATE: Series[pl.Float64] = pa.Field(nullable=True)

    class Config:
        name = "PanelTransformedSchema"
        strict = True
        ordered = False


# ── Post-validation helpers ────────────────────────────────────────────────


def validate_monthly_sparsity(df: pl.DataFrame) -> None:
    """Assert monthly series fill rate is between 2% and 8% (post-masking).

    Raises
    ------
    AssertionError
        If any monthly column falls outside the expected sparsity range.
    """
    for col in ["CPIAUCSL", "FEDFUNDS", "INDPRO", "UNRATE"]:
        n_total = len(df)
        n_filled = df[col].drop_nulls().len()
        pct = n_filled / n_total * 100
        assert 2.0 < pct < 8.0, (
            f"{col} fill rate {pct:.1f}% is outside the expected range [2%, 8%]. "
            "Check that pointwise masking (Rule 3 Step 4) was applied correctly."
        )


def validate_no_weekends(df: pl.DataFrame) -> None:
    """Assert the date spine contains no Saturday or Sunday dates.

    Raises
    ------
    AssertionError
        If any weekend date is found.
    """
    n_weekend = df.filter(pl.col("date").dt.weekday() > 5).height
    assert n_weekend == 0, (
        f"Found {n_weekend} weekend date(s) in the panel spine. "
        "Ensure build_spine excludes weekends."
    )


def validate_hy_oas_gap(df: pl.DataFrame) -> None:
    """Assert BAMLH0A0HYM2 has fewer than 200 nulls after 1997-01-01.

    Raises
    ------
    AssertionError
        If the null count exceeds the holiday-only threshold.
    """
    post_97 = df.filter(pl.col("date") >= pl.lit("1997-01-01").str.to_date())
    null_count = post_97["BAMLH0A0HYM2"].null_count()
    assert null_count < 200, (
        f"BAMLH0A0HYM2 has {null_count} unexpected nulls after 1997-01-01. "
        "Expected < 200 (only holidays). Check FRED pull for this series."
    )


# ── S3 upsert ──────────────────────────────────────────────────────────────


def upsert_s3(new_df: pl.DataFrame, key: str, bucket: str) -> int:
    """Incremental upsert: read existing Parquet, merge, write back.

    The merge deduplication strategy:
    - Keep all existing rows where  date < min(new_df["date"])
    - Append all rows from new_df
    - Sort by date

    Parameters
    ----------
    new_df:
        Freshly extracted/transformed rows for the pull window.
    key:
        S3 object key (e.g. ``"panel_raw.parquet"``).
    bucket:
        S3 bucket name.

    Returns
    -------
    int
        Total row count after upsert (existing + new, deduplicated).
    """
    try:
        obj = _s3.get_object(Bucket=bucket, Key=key)
        existing = pl.read_parquet(io.BytesIO(obj["Body"].read()))
        min_new_date = new_df["date"].min()
        existing = existing.filter(pl.col("date") < min_new_date)
        combined = pl.concat([existing, new_df]).sort("date")
    except _s3.exceptions.NoSuchKey:
        # First run — no existing file in S3
        combined = new_df.sort("date")

    buf = io.BytesIO()
    combined.write_parquet(buf)
    buf.seek(0)
    _s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())

    return len(combined)


# ── Orchestration ──────────────────────────────────────────────────────────


def validate_and_upload(
    df: pl.DataFrame,
    key: str,
    bucket: str,
    schema: Literal["raw", "transformed"],
) -> int:
    """Validate df with Pandera + post-checks, then upsert to S3.

    Parameters
    ----------
    df:
        DataFrame to validate and upload.
    key:
        S3 object key.
    bucket:
        S3 bucket name.
    schema:
        ``"raw"`` → :class:`PanelRawSchema`;
        ``"transformed"`` → :class:`PanelTransformedSchema`.

    Returns
    -------
    int
        Total row count after upsert.

    Raises
    ------
    pandera.errors.SchemaError
        If Pandera validation fails.
    AssertionError
        If any post-validation check fails.
    """
    schema_cls = PanelRawSchema if schema == "raw" else PanelTransformedSchema
    schema_cls.validate(df)

    validate_no_weekends(df)
    validate_hy_oas_gap(df)

    # Monthly sparsity check only applies to the transformed panel
    # (raw panel has monthly cols in level form, not masked yet)
    if schema == "transformed":
        validate_monthly_sparsity(df)

    return upsert_s3(df, key, bucket)
