"""Base loader abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLoader(ABC):
    """Abstract base class for all loaders.

    A loader receives the transformed data and persists it to a destination
    (S3, database, filesystem, etc.).
    """

    @abstractmethod
    def load(self, data: Any, key: str) -> None:  # noqa: ANN401
        """Persist *data* to the destination identified by *key*.

        Parameters
        ----------
        data:
            Transformed data ready to be stored.
        key:
            Destination identifier (e.g. an S3 object key or a file path).
        """
        raise NotImplementedError
