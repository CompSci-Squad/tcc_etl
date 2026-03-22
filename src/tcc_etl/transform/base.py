"""Base transformer abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import polars as pl


class BaseTransformer(ABC):
    """Abstract base class for all transformers.

    A transformer receives the raw output of an extractor and returns an
    iterator of :class:`polars.DataFrame` batches ready to be loaded.
    """

    @abstractmethod
    def transform(self, data: Any) -> Iterator[pl.DataFrame]:  # noqa: ANN401
        """Transform raw data into an iterator of DataFrame batches.

        Parameters
        ----------
        data:
            Raw data as returned by an extractor.

        Returns
        -------
        Iterator[polars.DataFrame]
            Transformed data batches ready to be loaded.
        """
        raise NotImplementedError
