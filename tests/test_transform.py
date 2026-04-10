from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from tcc_etl.transform import _tcode_expr, remove_outliers, transform_all




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



class TestTcodeExpr:
    def _eval(self, values: list[float], tcode: int, col: str = "x") -> list:
        lf = pl.DataFrame({"date": _dates(len(values)), col: values}).lazy()
        return lf.with_columns([_tcode_expr(col, tcode)]).collect()[col].to_list()

    def test_tcode1_identity(self) -> None:
        result = self._eval([1.0, 2.0, 3.0], 1)
        assert result == [1.0, 2.0, 3.0]

    def test_tcode2_first_diff(self) -> None:
        result = self._eval([10.0, 12.0, 15.0, 19.0], 2)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)
        assert result[3] == pytest.approx(4.0)

    def test_tcode3_second_diff(self) -> None:
        result = self._eval([1.0, 2.0, 4.0, 7.0], 3)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] is None or math.isnan(result[1])
        assert result[2] == pytest.approx(1.0)
        assert result[3] == pytest.approx(1.0)

    def test_tcode4_log(self) -> None:
        result = self._eval([1.0, math.e, -1.0], 4)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)
        assert result[2] is None or math.isnan(result[2])

    def test_tcode5_log_diff(self) -> None:
        result = self._eval([1.0, math.e, math.e**2], 5)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] == pytest.approx(1.0)
        assert result[2] == pytest.approx(1.0)

    def test_tcode6_log_second_diff(self) -> None:
        result = self._eval([1.0, math.e, math.e**2, math.e**4], 6)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] is None or math.isnan(result[1])
        assert result[2] == pytest.approx(0.0, abs=1e-9)
        assert result[3] == pytest.approx(1.0)

    def test_tcode7_pct_change_diff(self) -> None:
        result = self._eval([100.0, 110.0, 121.0, 133.1], 7)
        assert result[0] is None or math.isnan(result[0])
        assert result[1] is None or math.isnan(result[1])
        assert result[2] == pytest.approx(0.0, abs=1e-6)

    def test_unknown_tcode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="tcode desconhecido"):
            _tcode_expr("x", 99)

    def test_output_dtype_is_float64(self) -> None:
        lf = pl.DataFrame({"date": _dates(3), "x": [1.0, 2.0, 3.0]}).lazy()
        result = lf.with_columns([_tcode_expr("x", 2)]).collect()
        assert result["x"].dtype == pl.Float64

    def test_output_name_preserved(self) -> None:
        lf = pl.DataFrame({"date": _dates(3), "MYSERIES": [1.0, 2.0, 3.0]}).lazy()
        result = lf.with_columns([_tcode_expr("MYSERIES", 1)]).collect()
        assert "MYSERIES" in result.columns


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
