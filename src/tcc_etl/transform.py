from __future__ import annotations

import math

import polars as pl



def remove_outliers(
    lf: pl.LazyFrame,
    series_ids: list[str],
    k: float = 10.0,
) -> pl.LazyFrame:
    exprs = [
        pl.when(
            (pl.col(c) >= pl.col(c).median() - k * (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)))
            & (pl.col(c) <= pl.col(c).median() + k * (pl.col(c).quantile(0.75) - pl.col(c).quantile(0.25)))
        )
        .then(pl.col(c))
        .otherwise(None)
        .alias(c)
        for c in series_ids
    ]
    return lf.with_columns(exprs)


def _tcode_expr(c: str, tcode: int) -> pl.Expr:
    col = pl.col(c)
    log_c = pl.when(col > 0).then(col.log(math.e)).otherwise(None)
    if tcode == 1:
        return col.alias(c)
    elif tcode == 2:
        return col.diff().alias(c)
    elif tcode == 3:
        return col.diff().diff().alias(c)
    elif tcode == 4:
        return log_c.alias(c)
    elif tcode == 5:
        return log_c.diff().alias(c)
    elif tcode == 6:
        return log_c.diff().diff().alias(c)
    elif tcode == 7:
        shifted = col.shift(1)
        pct = pl.when(shifted != 0).then(col / shifted - 1.0).otherwise(None)
        return pct.diff().alias(c)
    else:
        raise ValueError(f"tcode desconhecido: {tcode} (serie: {c})")


def transform_all(
    lf: pl.LazyFrame,
    tcodes: dict[str, int],
    series_ids: list[str],
) -> pl.LazyFrame:
    exprs = [
        _tcode_expr(sid, tcodes[sid])
        for sid in series_ids
        if sid in tcodes
    ]
    if exprs:
        lf = lf.with_columns(exprs)

    # Drop the first 2 rows (consumed by .diff() in tcodes 3, 6).
    # NOTE: we deliberately do *not* backfill here. Imputation of leading
    # gaps (series whose recorded history starts after 1959-01-01) is
    # handled explicitly downstream by ``tcc_etl.imputation.EMFactorImputer``
    # so the policy is auditable and logged to S3 metadata.
    return lf.slice(2)

