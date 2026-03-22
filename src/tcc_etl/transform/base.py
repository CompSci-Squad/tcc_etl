"""Base transformer abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTransformer(ABC):
    """Abstract base class for all transformers.

    A transformer receives the raw output of an extractor and returns the
    processed data ready to be loaded.
    """

    @abstractmethod
    def transform(self, data: Any) -> Any:  # noqa: ANN401
        """Transform raw data into the desired output format.

        Parameters
        ----------
        data:
            Raw data as returned by an extractor.

        Returns
        -------
        Any
            Transformed data ready to be loaded.
        """
        raise NotImplementedError
