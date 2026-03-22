"""Base extractor abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseExtractor(ABC):
    """Abstract base class for all extractors.

    Every extractor must implement :py:meth:`extract` which returns the raw
    data as an arbitrary Python object that is then forwarded to the
    transformer.
    """

    @abstractmethod
    async def extract(self) -> Any:  # noqa: ANN401
        """Asynchronously extract raw data from the source.

        Returns
        -------
        Any
            Raw data.  The concrete type depends on the extractor implementation
            (e.g. ``list[dict]``, ``polars.DataFrame``, ``bytes``).
        """
        raise NotImplementedError
