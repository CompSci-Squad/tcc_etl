from __future__ import annotations

from datetime import date
from io import StringIO

import httpx
import polars as pl

_FRED_MD_BASE: str = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly"
)
_START_DATE: pl.Date = pl.date(1990, 1, 1)


def _fred_md_url(ref: date | None = None) -> str:
    today = ref or date.today()
    if today.month == 1:
        year, month = today.year - 1, 12
    else:
        year, month = today.year, today.month - 1
    return f"{_FRED_MD_BASE}/{year}-{month:02d}-md.csv"



async def fetch_fred_md() -> tuple[pl.LazyFrame, dict[str, int], list[str]]:
    header_line: str | None = None
    tcode_line: str | None = None
    data_lines: list[str] = []

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", _fred_md_url(), timeout=60.0) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if header_line is None:
                    header_line = line
                elif tcode_line is None:
                    tcode_line = line
                else:
                    data_lines.append(line)

    if header_line is None or tcode_line is None:
        raise ValueError("FRED-MD response contained no data")

    series_ids: list[str] = header_line.split(",")[1:]

    tcodes: dict[str, int] = {
        sid: int(float(tc))
        for sid, tc in zip(series_ids, tcode_line.split(",")[1:])
        if tc.strip() not in ("", "tcode")
    }

    data_text = "\n".join([header_line] + data_lines)

    lf = (
        pl.read_csv(
            StringIO(data_text),
            null_values=["", "NaN", "."],
            try_parse_dates=False,
        )
        .lazy()
        .with_columns(
            pl.col("sasdate").str.strptime(pl.Date, "%m/%d/%Y").alias("date")
        )
        .drop("sasdate")
        .filter(pl.col("date") >= _START_DATE)
    )

    return lf, tcodes, series_ids
