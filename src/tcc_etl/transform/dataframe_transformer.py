"""DataFrame transformer - cleans and reshapes tabular data with Polars.

Operates on a :class:`polars.LazyFrame` (or any source that can be converted
to one) and applies deduplication, null-row removal, and optional column
normalization via :mod:`tcc_etl.transform.numeric`.

Data is streamed out via :func:`polars.LazyFrame.collect_batches` so memory
usage stays bounded regardless of dataset size.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import polars as pl
from loguru import logger

from tcc_etl.config import get_settings
from tcc_etl.transform.base import BaseTransformer
from tcc_etl.transform.numeric import normalize_column


class DataFrameTransformer(BaseTransformer):
    """Transform tabular data stored as a :class:`polars.DataFrame`.

    Parameters
    ----------
    normalize_cols:
        Column names to min-max normalise using the JAX-accelerated kernel.
    drop_duplicates:
        Whether to drop duplicate rows (default: ``True``).
    dropna:
        Whether to drop rows that contain any ``null`` value (default: ``True``).
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
    def transform(self, data: Any) -> Iterator[pl.DataFrame]:  # noqa: ANN401
        """Apply cleaning and normalisation, yielding batches.

        Parameters
        ----------
        data:
            A :class:`polars.LazyFrame`, :class:`polars.DataFrame`,
            or a ``list[dict]`` of records.

        Yields
        ------
        polars.DataFrame
            Cleaned and transformed batches of size ``settings.batch_size``.

        Raises
        ------
        TypeError
            If *data* is not a supported type.
        """
        if isinstance(data, pl.LazyFrame):
            lf = data
        elif isinstance(data, pl.DataFrame):
            lf = data.lazy()
        elif isinstance(data, list):
            lf = pl.DataFrame(data).lazy()
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)!r}. "
                "Expected list[dict], polars.DataFrame, or polars.LazyFrame."
            )

        logger.info("Starting transformation with lazy operations")

        if self.dropna:
            lf = lf.drop_nulls()

        if self.drop_duplicates:
            lf = lf.unique()

        chunk_size: int = get_settings().batch_size
        batch_count = 0

        for batch in lf.collect_batches(chunk_size=chunk_size):
            # Apply per-column numeric normalization on each batch
            for col in self.normalize_cols:
                if col not in batch.columns:
                    logger.warning("Column '{}' not found - skipping normalisation", col)
                    continue
                try:
                    normalized = normalize_column(batch[col].to_list())
                    batch = batch.with_columns(pl.Series(col, normalized))
                    logger.debug("Normalised column '{}'", col)
                except (TypeError, ValueError) as exc:
                    logger.error("Failed to normalise column '{}': {}", col, exc)
                    raise

            logger.debug("Yielding batch {} - shape {}", batch_count, batch.shape)
            batch_count += 1
            yield batch

        logger.info("Transformation complete - yielded {} batch(es)", batch_count)
