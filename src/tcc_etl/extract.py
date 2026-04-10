from __future__ import annotations

import io
from datetime import date

import httpx
import polars as pl

_FRED_MD_BASE: str = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly"
)
_START_DATE: pl.Date = pl.date(1959, 1, 1)


def _fred_md_url(ref: date | None = None) -> str:
    today = ref or date.today()
    if today.month == 1:
        year, month = today.year - 1, 12
    else:
        year, month = today.year, today.month - 1
    return f"{_FRED_MD_BASE}/{year}-{month:02d}-md.csv"



async def fetch_fred_md() -> tuple[pl.LazyFrame, dict[str, int], list[str]]:
    buf = io.BytesIO()

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", _fred_md_url(), timeout=60.0) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=65_536):
                buf.write(chunk)

    raw: bytes = buf.getvalue()
    if not raw:
        raise ValueError("FRED-MD response contained no data")

    first_nl = raw.index(b"\n")
    second_nl = raw.index(b"\n", first_nl + 1)

    header_line = raw[:first_nl].decode("utf-8").rstrip("\r")
    tcode_line = raw[first_nl + 1 : second_nl].decode("utf-8").rstrip("\r")

    # Reconstruct CSV: header row + data rows only (tcode row excluded)
    csv_bytes = raw[: first_nl + 1] + raw[second_nl + 1 :]

    series_ids: list[str] = header_line.split(",")[1:]

    tcodes: dict[str, int] = {
        sid: int(float(tc))
        for sid, tc in zip(series_ids, tcode_line.split(",")[1:])
        if tc.strip() not in ("", "tcode")
    }

    lf = (
        pl.read_csv(
            io.BytesIO(csv_bytes),
            null_values=["", "NaN", "."],
            try_parse_dates=False,
            infer_schema_length=0,
            schema_overrides={sid: pl.Float64 for sid in series_ids},
        )
        .lazy()
        .with_columns(
            pl.col("sasdate").str.strptime(pl.Date, "%m/%d/%Y").alias("date")
        )
        .drop("sasdate")
        .filter(pl.col("date") >= _START_DATE)
    )

    return lf, tcodes, series_ids
