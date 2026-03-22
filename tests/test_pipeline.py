"""Tests for the ETL Pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tcc_etl.pipeline import Pipeline


def _make_pipeline(
    *,
    extract_return=None,
    transform_return=None,
) -> tuple[Pipeline, MagicMock, MagicMock, MagicMock]:
    extractor = MagicMock()
    extractor.extract.return_value = extract_return or {"raw": "data"}

    transformer = MagicMock()
    transformer.transform.return_value = transform_return or {"transformed": "data"}

    loader = MagicMock()

    pipeline = Pipeline(
        extractor=extractor,
        transformer=transformer,
        loader=loader,
        destination_key="output.json",
    )
    return pipeline, extractor, transformer, loader


class TestPipeline:
    def test_run_calls_all_steps(self) -> None:
        pipeline, extractor, transformer, loader = _make_pipeline()
        result = pipeline.run()

        extractor.extract.assert_called_once()
        transformer.transform.assert_called_once_with({"raw": "data"})
        loader.load.assert_called_once_with({"transformed": "data"}, "output.json")
        assert result == {"transformed": "data"}

    def test_run_propagates_extractor_error(self) -> None:
        pipeline, extractor, transformer, loader = _make_pipeline()
        extractor.extract.side_effect = RuntimeError("network down")

        with pytest.raises(RuntimeError, match="network down"):
            pipeline.run()

        transformer.transform.assert_not_called()
        loader.load.assert_not_called()

    def test_run_propagates_transformer_error(self) -> None:
        pipeline, _extractor, transformer, loader = _make_pipeline()
        transformer.transform.side_effect = ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            pipeline.run()

        loader.load.assert_not_called()

    def test_run_propagates_loader_error(self) -> None:
        pipeline, _extractor, _transformer, loader = _make_pipeline()
        loader.load.side_effect = OSError("S3 unavailable")

        with pytest.raises(OSError, match="S3 unavailable"):
            pipeline.run()
