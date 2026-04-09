from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from tcc_etl.transform import apply_tcode, remove_outliers, transform_all


def _series(values: list, name: str = "x") -> pl.Series:
    return pl.Series(name=name, values=values, dtype=pl.Float64)


def _dates(n: int) -> list[date]:
    from datetime import timedelta

    start = date(1990, 1, 1)
    return [start + timedelta(days=31 * i) for i in range(n)]



class TestRemoveOutliers:
    def test_extreme_value_replaced_with_null(self) -> None:
        lf = pl.DataFrame(
            {"date": _dates(6), "x": [1.0, 2.0, 3.0, 2.0, 1.0, 1000.0]}
        ).lazy()
        result = remove_outliers(lf, ["x"]).collect()
        assert result["x"][-1] is None

    def test_inlier_not_replaced(self) -> None:
        lf = pl.DataFrame(
            {"date": _dates(5), "x": [10.0, 11.0, 10.5, 10.8, 11.2]}
        ).lazy()
        result = remove_outliers(lf, ["x"]).collect()
        assert result["x"].null_count() == 0

    def test_returns_lazyframe(self, sample_raw_df: pl.DataFrame) -> None:
        result = remove_outliers(sample_raw_df.lazy(), ["SERIES1"])
        assert isinstance(result, pl.LazyFrame)

    def test_date_column_preserved(self, sample_raw_df: pl.DataFrame) -> None:
        result = remove_outliers(sample_raw_df.lazy(), ["SERIES1"]).collect()
        assert result["date"].null_count() == 0

    def test_custom_k_tighter_bound(self) -> None:
        lf = pl.DataFrame(
            {"date": _dates(6), "x": [1.0, 2.0, 3.0, 2.0, 1.0, 10.0]}
        ).lazy()
        result = remove_outliers(lf, ["x"], k=1.0).collect()
        assert result["x"][-1] is None



class TestApplyTcode:
    def test_tcode_1_identity(self) -> None:
        s = _series([1.0, 2.0, 3.0])
        result = apply_tcode(s, 1)
        assert list(result) == [1.0, 2.0, 3.0]

    def test_tcode_2_first_diff(self) -> None:
        s = _series([10.0, 12.0, 15.0, 19.0])
        result = apply_tcode(s, 2)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)
        assert result[3] == pytest.approx(4.0)

    def test_tcode_3_second_diff(self) -> None:
        s = _series([1.0, 2.0, 4.0, 7.0])
        result = apply_tcode(s, 3)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] is None or math.isnan(result[1])
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)

    def test_tcode_4_log(self) -> None:
        s = _series([1.0, math.e, -1.0])
        result = apply_tcode(s, 4)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] is None or math.isnan(result[2])

    def test_tcode_5_log_diff(self) -> None:
        s = _series([1.0, math.e, math.e**2])
        result = apply_tcode(s, 5)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)

    def test_tcode_6_log_second_diff(self) -> None:
        s = _series([1.0, math.e, math.e**2, math.e**4])
        result = apply_tcode(s, 6)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] is None or math.isnan(result[1])
        assert result[2] == pytest.approx(0.0, abs=1e-9)
        assert result[3] == pytest.approx(1.0)

    def test_tcode_7_pct_change_diff(self) -> None:
        s = _series([100.0, 110.0, 121.0, 133.1])
        result = apply_tcode(s, 7)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] is None or math.isnan(result[1])
        assert result[2] == pytest.approx(0.0, abs=1e-6)

    def test_unknown_tcode_raises_value_error(self) -> None:
        s = _series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="tcode desconhecido"):
            apply_tcode(s, 99)

    def test_output_is_polars_series(self) -> None:
        s = _series([1.0, 2.0, 3.0])
        result = apply_tcode(s, 1)
        assert isinstance(result, pl.Series)

    def test_output_dtype_is_float64(self) -> None:
        s = _series([1.0, 2.0, 3.0])
        result = apply_tcode(s, 2)
        assert result.dtype == pl.Float64

    def test_output_name_preserved(self) -> None:
        s = _series([1.0, 2.0], name="MYSERIES")
        result = apply_tcode(s, 1)
        assert result.name == "MYSERIES" or result.name == "MYSERIES".lower() or result.name == "MYSERIES" or result.name == s.name


# ---------------------------------------------------------------------------
# transform_all
# ---------------------------------------------------------------------------


class TestTransformAll:
    def _make_lf(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "date": _dates(5),
                "SERIES1": [100.0, 101.0, 102.0, 103.0, 104.0],
                "SERIES2": [200.0, 202.0, 205.0, 207.0, 210.0],
                "SERIES3": [50.0, 51.0, 52.0, 53.0, 54.0],
            }
        ).lazy()

    def test_drops_first_two_rows(self, sample_tcodes: dict[str, int]) -> None:
        lf = self._make_lf()
        result = transform_all(lf, sample_tcodes, ["SERIES1", "SERIES2", "SERIES3"]).collect()
        assert len(result) == 3

    def test_returns_lazyframe(self, sample_tcodes: dict[str, int]) -> None:
        lf = self._make_lf()
        result = transform_all(lf, sample_tcodes, ["SERIES1", "SERIES2", "SERIES3"])
        assert isinstance(result, pl.LazyFrame)

    def test_date_column_present(self, sample_tcodes: dict[str, int]) -> None:
        lf = self._make_lf()
        result = transform_all(lf, sample_tcodes, ["SERIES1", "SERIES2", "SERIES3"]).collect()
        assert "date" in result.columns

    def test_series_not_in_tcodes_skipped(self) -> None:
        lf = pl.DataFrame(
            {"date": _dates(4), "EXTRA": [1.0, 2.0, 3.0, 4.0]}
        ).lazy()
        result = transform_all(lf, {}, ["EXTRA"]).collect()
        assert "EXTRA" in result.columns
