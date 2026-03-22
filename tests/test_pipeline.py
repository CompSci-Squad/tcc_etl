"""Tests for the ETL Pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from tcc_etl.pipeline import Pipeline


def _make_pipeline(
    *,
    extract_return=None,
    batches: list | None = None,
) -> tuple[Pipeline, AsyncMock, MagicMock, AsyncMock]:
    extractor = AsyncMock()
    extractor.extract.return_value = extract_return or {"raw": "data"}

    default_batch = pl.DataFrame({"transformed": [1, 2]})
    transformer = MagicMock()
    transformer.transform.return_value = iter(batches if batches is not None else [default_batch])

    loader = AsyncMock()

    pipeline = Pipeline(
        extractor=extractor,
        transformer=transformer,
        loader=loader,
        destination_key="output.parquet",
    )
    return pipeline, extractor, transformer, loader


class TestPipeline:
    async def test_run_calls_all_steps(self) -> None:
        batch = pl.DataFrame({"transformed": [1, 2]})
        pipeline, extractor, transformer, loader = _make_pipeline(batches=[batch])

        batch_count = await pipeline.run()

        extractor.extract.assert_awaited_once()
        transformer.transform.assert_called_once_with({"raw": "data"})
        loader.load.assert_awaited_once_with(batch, "output_part0000.parquet")
        assert batch_count == 1

    async def test_run_multiple_batches(self) -> None:
        batches = [pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [2]}), pl.DataFrame({"a": [3]})]
        pipeline, _, _, loader = _make_pipeline(batches=batches)

        batch_count = await pipeline.run()

        assert batch_count == 3
        keys = [call.args[1] for call in loader.load.await_args_list]
        assert keys == ["output_part0000.parquet", "output_part0001.parquet", "output_part0002.parquet"]

    async def test_run_propagates_extractor_error(self) -> None:
        pipeline, extractor, transformer, loader = _make_pipeline()
        extractor.extract.side_effect = RuntimeError("network down")

        with pytest.raises(RuntimeError, match="network down"):
            await pipeline.run()

        transformer.transform.assert_not_called()
        loader.load.assert_not_awaited()

    async def test_run_propagates_transformer_error(self) -> None:
        pipeline, _extractor, transformer, loader = _make_pipeline()
        transformer.transform.side_effect = ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            await pipeline.run()

        loader.load.assert_not_awaited()

    async def test_run_propagates_loader_error(self) -> None:
        pipeline, _extractor, _transformer, loader = _make_pipeline()
        loader.load.side_effect = OSError("S3 unavailable")

        with pytest.raises(OSError, match="S3 unavailable"):
            await pipeline.run()

    def test_batch_key_format(self) -> None:
        assert Pipeline._batch_key("output", ".parquet", 0) == "output_part0000.parquet"
        assert Pipeline._batch_key("output", ".parquet", 42) == "output_part0042.parquet"
        assert Pipeline._batch_key("data", ".csv", 9999) == "data_part9999.csv"
