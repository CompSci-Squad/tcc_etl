"""Extract layer -- download the FRED-MD monthly dataset via async HTTP streaming.

Public API:
- fetch_fred_md() -> tuple[pl.LazyFrame, dict[str, int], list[str]]

The response is consumed line-by-line with httpx streaming so the header and
tcode rows are parsed as soon as they arrive.  Data rows are accumulated in a
list (unavoidable -- columnar stats such as median/IQR and ADF require the full
time series before any transformation can run).
"""

from __future__ import annotations

from io import StringIO

import httpx
import polars as pl

# -- Constants -----------------------------------------------------------------

FRED_MD_URL: str = (
    "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
)
_START_DATE: pl.Date = pl.date(1990, 1, 1)


# -- Public API ----------------------------------------------------------------


async def fetch_fred_md() -> tuple[pl.LazyFrame, dict[str, int], list[str]]:
    """Stream the current FRED-MD CSV and return a LazyFrame plus metadata.

    Streaming protocol:
    - Line 0 (header) and Line 1 (tcode) are parsed the moment each arrives.
    - Data lines (Line 2+) are collected into a list so that the full time
      series is available for the downstream columnar transforms.
    - An HTTP error status raises httpx.HTTPStatusError before any data is
      consumed.
    """
    header_line: str | None = None
    tcode_line: str | None = None
    data_lines: list[str] = []

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", FRED_MD_URL, timeout=60.0) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if header_line is None:
                    # Line 0: sasdate,SERIES1,SERIES2,...
                    header_line = line
                elif tcode_line is None:
                    # Line 1: tcode,1,5,2,...  -- parsed immediately
                    tcode_line = line
                else:
                    # Lines 2+: monthly observations
                    data_lines.append(line)

    if header_line is None or tcode_line is None:
        raise ValueError("FRED-MD response contained no data")

    series_ids: list[str] = header_line.split(",")[1:]

    tcodes: dict[str, int] = {
        sid: int(float(tc))
        for sid, tc in zip(series_ids, tcode_line.split(",")[1:])
        if tc.strip() not in ("", "tcode")
    }

    # Rebuild a clean CSV (header + data only, no tcode row)
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
