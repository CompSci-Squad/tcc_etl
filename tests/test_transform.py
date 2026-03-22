"""Tests for the transform package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcc_etl.transform import DataFrameTransformer, clip_column, normalize_column


class TestNumericHelpers:
    def test_normalize_column_basic(self) -> None:
        arr = np.array([0.0, 5.0, 10.0])
        result = normalize_column(arr)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_normalize_column_all_same(self) -> None:
        arr = np.array([3.0, 3.0, 3.0])
        result = normalize_column(arr)
        # When range is 0 the original array is returned unchanged
        np.testing.assert_array_equal(result, arr)

    def test_clip_column(self) -> None:
        arr = np.array([-1.0, 0.5, 2.0])
        result = clip_column(arr, 0.0, 1.0)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)


class TestDataFrameTransformer:
    def test_drop_duplicates(self, sample_dataframe: pd.DataFrame) -> None:
        transformer = DataFrameTransformer(drop_duplicates=True, dropna=False)
        result = transformer.transform(sample_dataframe)
        assert len(result) == 3  # one duplicate removed

    def test_dropna(self) -> None:
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
        transformer = DataFrameTransformer(drop_duplicates=False, dropna=True)
        result = transformer.transform(df)
        assert len(result) == 2

    def test_normalize_cols(self, sample_dataframe: pd.DataFrame) -> None:
        transformer = DataFrameTransformer(
            normalize_cols=["value"],
            drop_duplicates=True,
            dropna=False,
        )
        result = transformer.transform(sample_dataframe)
        assert result["value"].min() == pytest.approx(0.0)
        assert result["value"].max() == pytest.approx(1.0)

    def test_normalize_missing_col_is_skipped(self, sample_dataframe: pd.DataFrame) -> None:
        transformer = DataFrameTransformer(
            normalize_cols=["nonexistent"],
            drop_duplicates=False,
            dropna=False,
        )
        # Should not raise
        result = transformer.transform(sample_dataframe)
        assert list(result.columns) == list(sample_dataframe.columns)

    def test_accepts_list_of_dicts(self, sample_records: list[dict]) -> None:
        transformer = DataFrameTransformer()
        result = transformer.transform(sample_records)
        assert isinstance(result, pd.DataFrame)
