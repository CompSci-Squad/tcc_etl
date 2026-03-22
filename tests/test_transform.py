"""Tests for the transform package."""

from __future__ import annotations

import pytest
import polars as pl

from tcc_etl.transform import DataFrameTransformer, clip_column, normalize_column


class TestNumericHelpers:
    def test_normalize_column_basic(self) -> None:
        result = normalize_column([0.0, 5.0, 10.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_normalize_column_all_same(self) -> None:
        result = normalize_column([3.0, 3.0, 3.0])
        # When range is 0 every value maps to 0.0
        assert all(v == pytest.approx(0.0) for v in result)

    def test_normalize_column_empty(self) -> None:
        assert normalize_column([]) == []

    def test_clip_column(self) -> None:
        result = clip_column([-1.0, 0.5, 2.0], 0.0, 1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_clip_column_empty(self) -> None:
        assert clip_column([], 0.0, 1.0) == []


class TestDataFrameTransformer:
    def _collect(self, transformer: DataFrameTransformer, data) -> pl.DataFrame:
        """Collect all batches from the transformer into a single DataFrame."""
        batches = list(transformer.transform(data))
        assert len(batches) > 0, "Transformer yielded no batches"
        return pl.concat(batches)

    def test_drop_duplicates(self, sample_dataframe: pl.DataFrame) -> None:
        transformer = DataFrameTransformer(drop_duplicates=True, dropna=False)
        result = self._collect(transformer, sample_dataframe)
        assert len(result) == 3  # one duplicate removed

    def test_dropna(self) -> None:
        df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
        transformer = DataFrameTransformer(drop_duplicates=False, dropna=True)
        result = self._collect(transformer, df)
        assert len(result) == 2

    def test_normalize_cols(self, sample_dataframe: pl.DataFrame) -> None:
        transformer = DataFrameTransformer(
            normalize_cols=["value"],
            drop_duplicates=True,
            dropna=False,
        )
        result = self._collect(transformer, sample_dataframe)
        assert result["value"].min() == pytest.approx(0.0)
        assert result["value"].max() == pytest.approx(1.0)

    def test_normalize_missing_col_is_skipped(self, sample_dataframe: pl.DataFrame) -> None:
        transformer = DataFrameTransformer(
            normalize_cols=["nonexistent"],
            drop_duplicates=False,
            dropna=False,
        )
        # Should not raise
        result = self._collect(transformer, sample_dataframe)
        assert result.columns == sample_dataframe.columns

    def test_accepts_list_of_dicts(self, sample_records: list[dict]) -> None:
        transformer = DataFrameTransformer()
        result = self._collect(transformer, sample_records)
        assert isinstance(result, pl.DataFrame)

    def test_accepts_lazyframe(self, sample_dataframe: pl.DataFrame) -> None:
        transformer = DataFrameTransformer(drop_duplicates=False, dropna=False)
        result = self._collect(transformer, sample_dataframe.lazy())
        assert isinstance(result, pl.DataFrame)

    def test_unsupported_type_raises(self) -> None:
        transformer = DataFrameTransformer()
        with pytest.raises(TypeError, match="Unsupported data type"):
            list(transformer.transform(b"raw bytes"))

    def test_empty_dataframe_yields_no_rows(self) -> None:
        df = pl.DataFrame({"a": [], "b": []})
        transformer = DataFrameTransformer(drop_duplicates=False, dropna=False)
        batches = list(transformer.transform(df))
        total = sum(len(b) for b in batches)
        assert total == 0
