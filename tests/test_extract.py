from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import polars as pl
import pytest

from tcc_etl.extract import _fred_md_url, fetch_fred_md


class TestFetchFredMd:
    async def test_returns_lazyframe(self, fred_md_stream_mock: AsyncMock) -> None:
        lf, _, _ = await fetch_fred_md()
        assert isinstance(lf, pl.LazyFrame)

    async def test_tcodes_parsed_correctly(
        self, fred_md_stream_mock: AsyncMock, sample_tcodes: dict[str, int]
    ) -> None:
        _, tcodes, _ = await fetch_fred_md()
        assert tcodes == sample_tcodes

    async def test_series_ids_order_preserved(
        self, fred_md_stream_mock: AsyncMock, sample_series_ids: list[str]
    ) -> None:
        _, _, series_ids = await fetch_fred_md()
        assert series_ids == sample_series_ids

    async def test_date_filter_applied(self, fred_md_stream_mock: AsyncMock) -> None:
        lf, _, _ = await fetch_fred_md()
        assert len(lf.collect()) == 5

    async def test_date_filter_removes_pre_1990(self) -> None:
        from collections.abc import AsyncGenerator
        from unittest.mock import AsyncMock, MagicMock

        lines = [
            "sasdate,SERIES1",
            "tcode,1",
            "1/1/1989,99.0",
            "1/1/1990,100.0",
            "2/1/1990,101.0",
        ]

        class _FakeStream:
            async def __aenter__(self) -> "_FakeStream":
                return self
            async def __aexit__(self, *a: object) -> None:
                pass
            async def aiter_lines(self) -> AsyncGenerator[str, None]:
                for ln in lines:
                    yield ln
            def raise_for_status(self) -> None:
                pass

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=_FakeStream())
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_client
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("tcc_etl.extract.httpx.AsyncClient", mock_cls):
            lf, _, _ = await fetch_fred_md()

        df = lf.collect()
        assert len(df) == 2
        assert df["date"].min() == date(1990, 1, 1)

    async def test_date_dtype_is_pl_date(self, fred_md_stream_mock: AsyncMock) -> None:
        lf, _, _ = await fetch_fred_md()
        assert lf.collect_schema()["date"] == pl.Date

    async def test_tcode_row_not_in_data(self, fred_md_stream_mock: AsyncMock) -> None:
        lf, _, _ = await fetch_fred_md()
        df = lf.collect()
        assert df["SERIES1"].min() > 90.0

    async def test_http_error_propagates(self) -> None:
        fake_stream = AsyncMock()
        fake_stream.__aenter__.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=fake_stream)
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_client
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("tcc_etl.extract.httpx.AsyncClient", mock_cls):
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_fred_md()

    async def test_uses_prior_month_url(self, fred_md_stream_mock: AsyncMock) -> None:
        await fetch_fred_md()
        fred_md_stream_mock.stream.assert_called_once()
        url = fred_md_stream_mock.stream.call_args[0][1]
        assert url == _fred_md_url()
        assert url.endswith("-md.csv")


class TestFredMdUrl:
    def test_april_returns_march(self) -> None:
        assert _fred_md_url(date(2026, 4, 9)).endswith("2026-03-md.csv")

    def test_january_returns_december_prior_year(self) -> None:
        assert _fred_md_url(date(2026, 1, 15)).endswith("2025-12-md.csv")

    def test_december_returns_november(self) -> None:
        assert _fred_md_url(date(2025, 12, 1)).endswith("2025-11-md.csv")

    def test_url_contains_base_path(self) -> None:
        url = _fred_md_url(date(2026, 4, 1))
        assert "fred-md/monthly" in url
