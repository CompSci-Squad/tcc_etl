"""File extractor - reads data from a local file."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger

from tcc_etl.extract.base import BaseExtractor


class FileExtractor(BaseExtractor):
    """Extract data from a local file.

    Supported formats: ``json``, ``csv``, and raw ``bytes`` (any other
    extension).

    Parameters
    ----------
    path:
        Path to the source file.
    encoding:
        File encoding (default: ``utf-8``).
    """

    def __init__(self, path: str | Path, *, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    async def extract(self) -> Any:  # noqa: ANN401
        """Read and parse the file.

        Returns
        -------
        Any
            Parsed content: ``dict``/``list`` for JSON, ``list[dict]`` for CSV,
            ``bytes`` for any other format.
        """
        logger.info("Extracting data from file {}", self.path)
        suffix = self.path.suffix.lower()

        if suffix == ".json":
            data = json.loads(self.path.read_text(encoding=self.encoding))
            logger.debug("Extracted JSON payload from {}", self.path)
            return data

        if suffix == ".csv":
            with self.path.open(encoding=self.encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            logger.debug("Extracted {} CSV rows from {}", len(rows), self.path)
            return rows

        raw = self.path.read_bytes()
        logger.debug("Extracted {} bytes from {}", len(raw), self.path)
        return raw
