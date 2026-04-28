"""EM-PCA imputation for macro panels (Stock & Watson, 2002).

Iteratively reconstructs missing entries via truncated SVD on the
standardized panel until the Frobenius-norm change falls below ``tol``.

References
----------
Stock, J.H., Watson, M.W. (2002). "Macroeconomic Forecasting Using Diffusion
Indexes", JBES 20(2). McCracken & Ng (2016) FRED-MD documentation §3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class ImputationReport:
    kept_series: list[str]
    dropped_series: list[str]
    n_iter: int
    converged: bool
    final_delta: float
    frac_imputed: float
    k_factors: int

    def to_dict(self) -> dict:
        return {
            "kept_series": self.kept_series,
            "dropped_series": self.dropped_series,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "final_delta": float(self.final_delta),
            "frac_imputed": float(self.frac_imputed),
            "k_factors": int(self.k_factors),
            "n_kept": len(self.kept_series),
            "n_dropped": len(self.dropped_series),
        }


@dataclass
class EMFactorImputer:
    """Iterative truncated-SVD imputer for (T, N) panels with NaNs.

    Parameters
    ----------
    k : int
        Number of latent factors (default 8 — McCracken & Ng FRED-MD).
    max_iter : int
        EM iterations cap.
    tol : float
        Convergence threshold on relative Frobenius-norm delta.
    max_missing_frac : float
        Series with more than this fraction missing are dropped before
        imputation (no extrapolation beyond data support).
    """

    k: int = 8
    max_iter: int = 50
    tol: float = 1e-4
    max_missing_frac: float = 0.33

    _means: np.ndarray = field(default=None, init=False, repr=False)
    _stds: np.ndarray = field(default=None, init=False, repr=False)
    report: ImputationReport | None = field(default=None, init=False)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        self._means = np.nanmean(X, axis=0)
        self._stds = np.nanstd(X, axis=0, ddof=0)
        self._stds = np.where(self._stds < 1e-12, 1.0, self._stds)
        return (X - self._means) / self._stds

    def _destandardize(self, Z: np.ndarray) -> np.ndarray:
        return Z * self._stds + self._means

    def _select_columns(
        self, X: np.ndarray, columns: list[str]
    ) -> tuple[np.ndarray, list[str], list[str]]:
        miss_frac = np.isnan(X).mean(axis=0)
        keep_mask = miss_frac <= self.max_missing_frac
        kept = [c for c, m in zip(columns, keep_mask) if m]
        dropped = [c for c, m in zip(columns, keep_mask) if not m]
        return X[:, keep_mask], kept, dropped

    def fit_transform_panel(
        self, X: np.ndarray, columns: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Drop sparse cols, EM-impute the rest, return (X_imputed, kept_cols)."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (T, N); got shape {X.shape}")

        X_kept, kept, dropped = self._select_columns(X, columns)

        if X_kept.size == 0:
            self.report = ImputationReport(
                kept_series=kept,
                dropped_series=dropped,
                n_iter=0,
                converged=True,
                final_delta=0.0,
                frac_imputed=0.0,
                k_factors=0,
            )
            return X_kept, kept

        T, N = X_kept.shape
        k_eff = max(1, min(self.k, min(T, N) - 1))

        Z = self._standardize(X_kept)
        mask = np.isnan(Z)
        n_missing = int(mask.sum())
        Z_filled = np.where(mask, 0.0, Z)  # mean-imputation in standardized space

        delta_rel: float = float("inf")
        converged = False
        n_iter = 0
        for it in range(1, self.max_iter + 1):
            U, s, Vt = np.linalg.svd(Z_filled, full_matrices=False)
            Z_hat = (U[:, :k_eff] * s[:k_eff]) @ Vt[:k_eff, :]

            denom = np.linalg.norm(Z_filled[mask]) + 1e-12
            delta_rel = float(np.linalg.norm(Z_hat[mask] - Z_filled[mask]) / denom)

            Z_filled = np.where(mask, Z_hat, Z_filled)
            n_iter = it
            if delta_rel < self.tol:
                converged = True
                break

        X_imputed = self._destandardize(Z_filled)
        self.report = ImputationReport(
            kept_series=kept,
            dropped_series=dropped,
            n_iter=n_iter,
            converged=converged,
            final_delta=delta_rel,
            frac_imputed=n_missing / max(T * N, 1),
            k_factors=k_eff,
        )
        return X_imputed, kept


def impute_lazyframe(
    lf: pl.LazyFrame,
    series_ids: list[str],
    *,
    k: int = 8,
    max_iter: int = 50,
    tol: float = 1e-4,
    max_missing_frac: float = 0.33,
) -> tuple[pl.LazyFrame, ImputationReport]:
    """Apply EM-PCA imputation to series columns of a Polars LazyFrame.

    Non-series columns (e.g. ``date``) are passed through unchanged.
    """
    df = lf.collect()
    schema_cols = df.columns
    present = [c for c in series_ids if c in schema_cols]
    if not present:
        imputer = EMFactorImputer(
            k=k, max_iter=max_iter, tol=tol, max_missing_frac=max_missing_frac
        )
        imputer.report = ImputationReport([], [], 0, True, 0.0, 0.0, 0)
        return lf, imputer.report

    X = df.select(present).to_numpy()
    imputer = EMFactorImputer(
        k=k, max_iter=max_iter, tol=tol, max_missing_frac=max_missing_frac
    )
    X_imp, kept = imputer.fit_transform_panel(X, present)

    other_cols = [c for c in schema_cols if c not in present]
    out = df.select(other_cols).hstack(
        pl.DataFrame({c: X_imp[:, i] for i, c in enumerate(kept)})
    )
    return out.lazy(), imputer.report
