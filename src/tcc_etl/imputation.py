"""EM-PCA imputation for macro panels (Stock & Watson, 2002).

Iteratively reconstructs missing entries via truncated SVD on the
standardized panel until the Frobenius-norm change falls below ``tol``.

The imputer also produces a Boolean **mask** of shape ``(T, len(kept_series))``
that records which cells were originally NaN (and therefore imputed). The mask
is used downstream:

- ``tcc_etl`` writes it as a sidecar parquet alongside the transformed panel.
- ``tcc_ai.WindowDataset`` filters training windows whose **target** cell is
  imputed, preventing the autoencoder from learning to reproduce values that
  the imputer itself produced (target-only validity policy).

For each series we further distinguish two kinds of original missingness:

- **Leading** — NaNs before the first ever observation of the series (e.g.
  ``VXOCLSx`` is missing pre-1986 because the VIX did not exist).
- **Internal** — NaNs between observations (publication delays, revisions).

Both are imputed identically, but reported separately so downstream consumers
(and the thesis) can defend the trade-off.

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
    frac_imputed_leading: float
    frac_imputed_internal: float
    k_factors: int

    def to_dict(self) -> dict:
        return {
            "kept_series": self.kept_series,
            "dropped_series": self.dropped_series,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "final_delta": float(self.final_delta),
            "frac_imputed": float(self.frac_imputed),
            "frac_imputed_leading": float(self.frac_imputed_leading),
            "frac_imputed_internal": float(self.frac_imputed_internal),
            "k_factors": int(self.k_factors),
            "n_kept": len(self.kept_series),
            "n_dropped": len(self.dropped_series),
        }


def _split_leading_internal(mask: np.ndarray) -> tuple[int, int]:
    """Count leading vs internal missing cells in a (T, N) Boolean mask."""
    if mask.size == 0:
        return 0, 0
    obs = ~mask
    has_seen = np.cumsum(obs, axis=0) > 0
    leading = int(np.sum(mask & ~has_seen))
    internal = int(np.sum(mask & has_seen))
    return leading, internal


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
    max_missing_frac: float = 0.5

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
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Drop sparse cols, EM-impute the rest.

        Returns
        -------
        X_imputed : (T, len(kept)) ndarray
        kept_columns : list[str]
        mask : (T, len(kept)) Boolean ndarray; True where original was NaN.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (T, N); got shape {X.shape}")

        X_kept, kept, dropped = self._select_columns(X, columns)
        mask = np.isnan(X_kept)

        if X_kept.size == 0:
            self.report = ImputationReport(
                kept_series=kept,
                dropped_series=dropped,
                n_iter=0,
                converged=True,
                final_delta=0.0,
                frac_imputed=0.0,
                frac_imputed_leading=0.0,
                frac_imputed_internal=0.0,
                k_factors=0,
            )
            return X_kept, kept, mask

        T, N = X_kept.shape
        k_eff = max(1, min(self.k, min(T, N) - 1))

        Z = self._standardize(X_kept)
        n_missing = int(mask.sum())
        Z_filled = np.where(mask, 0.0, Z)

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

        leading, internal = _split_leading_internal(mask)
        cells = max(T * N, 1)
        self.report = ImputationReport(
            kept_series=kept,
            dropped_series=dropped,
            n_iter=n_iter,
            converged=converged,
            final_delta=delta_rel,
            frac_imputed=n_missing / cells,
            frac_imputed_leading=leading / cells,
            frac_imputed_internal=internal / cells,
            k_factors=k_eff,
        )
        return X_imputed, kept, mask


def impute_lazyframe(
    lf: pl.LazyFrame,
    series_ids: list[str],
    *,
    k: int = 8,
    max_iter: int = 50,
    tol: float = 1e-4,
    max_missing_frac: float = 0.5,
) -> tuple[pl.LazyFrame, pl.LazyFrame, ImputationReport]:
    """Apply EM-PCA imputation to series columns of a Polars LazyFrame.

    Returns
    -------
    panel_lf : LazyFrame
        ``date`` + imputed series columns (kept only).
    mask_lf : LazyFrame
        ``date`` + Boolean columns (one per kept series). ``True`` means the
        cell was originally NaN and was filled by EM-PCA.
    report : ImputationReport
    """
    df = lf.collect()
    schema_cols = df.columns
    present = [c for c in series_ids if c in schema_cols]

    if not present:
        empty_report = ImputationReport(
            kept_series=[],
            dropped_series=[],
            n_iter=0,
            converged=True,
            final_delta=0.0,
            frac_imputed=0.0,
            frac_imputed_leading=0.0,
            frac_imputed_internal=0.0,
            k_factors=0,
        )
        date_only = df.select([c for c in schema_cols if c == "date"])
        return date_only.lazy(), date_only.lazy(), empty_report

    X = df.select(present).to_numpy()
    imputer = EMFactorImputer(
        k=k, max_iter=max_iter, tol=tol, max_missing_frac=max_missing_frac
    )
    X_imp, kept, mask = imputer.fit_transform_panel(X, present)

    other_cols = [c for c in schema_cols if c not in present]
    panel_df = df.select(other_cols).hstack(
        pl.DataFrame({c: X_imp[:, i] for i, c in enumerate(kept)})
    )

    mask_other_cols = [c for c in other_cols if c == "date"]
    mask_df = df.select(mask_other_cols).hstack(
        pl.DataFrame({c: mask[:, i] for i, c in enumerate(kept)})
    )

    return panel_df.lazy(), mask_df.lazy(), imputer.report
