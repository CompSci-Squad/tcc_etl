"""HTTP extractor - fetches data from a remote URL."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import urlopen

from loguru import logger

from tcc_etl.extract.base import BaseExtractor


class HttpExtractor(BaseExtractor):
    """Extract data from an HTTP/HTTPS endpoint.

    Parameters
    ----------
    url:
        The URL to fetch.
    timeout:
        Request timeout in seconds (default: 30).
    as_json:
        When ``True`` the response body is parsed as JSON and returned as a
        Python object.  When ``False`` the raw bytes are returned.
    """

    def __init__(self, url: str, *, timeout: int = 30, as_json: bool = True) -> None:
        self.url = url
        self.timeout = timeout
        self.as_json = as_json

    def extract(self) -> Any:  # noqa: ANN401
        """Fetch data from :attr:`url`.

        Returns
        -------
        Any
            Parsed JSON object when ``as_json=True``, otherwise raw ``bytes``.
        """
        logger.info("Extracting data from {}", self.url)
        with urlopen(self.url, timeout=self.timeout) as response:  # noqa: S310
            raw: bytes = response.read()

        if self.as_json:
            data = json.loads(raw)
            logger.debug("Extracted {} records", len(data) if isinstance(data, list) else 1)
            return data

        logger.debug("Extracted {} bytes", len(raw))
        return raw
