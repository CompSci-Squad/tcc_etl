"""DataFrame transformer - cleans and reshapes tabular data.

Operates on a ``pandas.DataFrame`` (or a ``list[dict]`` which is automatically
converted) and applies column normalisation, type coercion, and optional
numeric acceleration via :mod:`tcc_etl.transform.numeric`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from tcc_etl.transform.base import BaseTransformer
from tcc_etl.transform.numeric import normalize_column


class DataFrameTransformer(BaseTransformer):
    """Transform tabular data stored as a ``pandas.DataFrame``.

    Parameters
    ----------
    normalize_cols:
        Column names to min-max normalise using the Numba-accelerated kernel.
    drop_duplicates:
        Whether to drop duplicate rows (default: ``True``).
    dropna:
        Whether to drop rows that contain any ``NaN`` value (default: ``True``).
    """

    def __init__(
        self,
        *,
        normalize_cols: list[str] | None = None,
        drop_duplicates: bool = True,
        dropna: bool = True,
    ) -> None:
        self.normalize_cols = normalize_cols or []
        self.drop_duplicates = drop_duplicates
        self.dropna = dropna

    # ------------------------------------------------------------------
    def transform(self, data: Any) -> pd.DataFrame:  # noqa: ANN401
        """Apply cleaning and normalisation steps to *data*.

        Parameters
        ----------
        data:
            A ``pd.DataFrame`` or a ``list[dict]`` of records.

        Returns
        -------
        pd.DataFrame
            The cleaned and transformed frame.
        """
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        logger.info("Transforming DataFrame with shape {}", df.shape)

        if self.dropna:
            before = len(df)
            df = df.dropna()
            dropped = before - len(df)
            if dropped:
                logger.debug("Dropped {} rows containing NaN", dropped)

        if self.drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            dropped = before - len(df)
            if dropped:
                logger.debug("Dropped {} duplicate rows", dropped)

        for col in self.normalize_cols:
            if col not in df.columns:
                logger.warning("Column '{}' not found - skipping normalisation", col)
                continue
            df[col] = normalize_column(df[col].to_numpy(dtype=float))
            logger.debug("Normalised column '{}'", col)

        logger.info("Transformation complete - output shape {}", df.shape)
        return df
