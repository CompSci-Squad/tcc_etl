from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tcc_etl.imputation import EMFactorImputer, impute_lazyframe


@pytest.fixture()
def synthetic_factor_panel() -> tuple[np.ndarray, np.ndarray]:
    """Generate (T=240, N=20) panel from k=3 latent factors + noise."""
    rng = np.random.default_rng(42)
    T, N, k = 240, 20, 3
    F = rng.standard_normal((T, k))
    L = rng.standard_normal((k, N))
    X_full = F @ L + 0.05 * rng.standard_normal((T, N))
    X_miss = X_full.copy()
    # Simulate "late starters": columns 0-3 missing first 60 months
    X_miss[:60, :4] = np.nan
    return X_full, X_miss


class TestEMFactorImputer:
    def test_recovers_factor_structure(self, synthetic_factor_panel):
        X_full, X_miss = synthetic_factor_panel
        imp = EMFactorImputer(k=3, max_iter=100, tol=1e-6, max_missing_frac=0.5)
        X_hat, kept, mask = imp.fit_transform_panel(
            X_miss, [f"S{i}" for i in range(X_miss.shape[1])]
        )
        assert len(kept) == X_miss.shape[1]
        assert imp.report.converged
        assert mask.shape == X_miss.shape
        np.testing.assert_array_equal(mask, np.isnan(X_miss))
        mae = np.mean(np.abs(X_hat[mask] - X_full[mask]))
        assert mae < 0.5

    def test_drops_high_missingness(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 5))
        X[:90, 0] = np.nan  # 75% missing → above 0.33 default
        imp = EMFactorImputer(k=2, max_missing_frac=0.33)
        _, kept, mask = imp.fit_transform_panel(X, [f"S{i}" for i in range(5)])
        assert "S0" not in kept
        assert imp.report.dropped_series == ["S0"]
        assert mask.shape == (120, 4)
        assert mask.sum() == 0

    def test_no_missing_passthrough(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 4))
        imp = EMFactorImputer(k=2)
        X_hat, _, mask = imp.fit_transform_panel(X, ["A", "B", "C", "D"])
        assert imp.report.frac_imputed == 0.0
        assert imp.report.frac_imputed_leading == 0.0
        assert imp.report.frac_imputed_internal == 0.0
        assert mask.sum() == 0
        np.testing.assert_allclose(X_hat, X, rtol=1e-10, atol=1e-10)

    def test_leading_internal_split(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((100, 3))
        # Series 0: leading NaNs (rows 0-19)
        X[:20, 0] = np.nan
        # Series 1: internal NaNs (rows 50, 60, 70)
        X[[50, 60, 70], 1] = np.nan
        imp = EMFactorImputer(k=1, max_missing_frac=0.5)
        _, _, _ = imp.fit_transform_panel(X, ["A", "B", "C"])
        rep = imp.report
        # 20 leading + 3 internal of 300 cells
        assert rep.frac_imputed_leading == pytest.approx(20 / 300)
        assert rep.frac_imputed_internal == pytest.approx(3 / 300)


class TestImputeLazyframe:
    def test_preserves_date_column(self):
        from datetime import date

        df = pl.DataFrame(
            {
                "date": [date(1990, m, 1) for m in range(1, 13)],
                "A": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "B": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0],
            }
        )
        out_lf, mask_lf, report = impute_lazyframe(df.lazy(), ["A", "B"], k=1)
        out = out_lf.collect()
        mask = mask_lf.collect()
        assert "date" in out.columns
        assert "date" in mask.columns
        assert out["A"].null_count() == 0
        assert report.frac_imputed > 0
        # Mask shape matches panel shape (excluding date)
        assert mask.shape == out.shape
        # Cell (1, A) was the only original NaN → mask True there
        assert mask["A"][1] is True
        assert mask["A"][0] is False
