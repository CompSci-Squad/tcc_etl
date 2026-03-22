"""HTTP extractor - fetches data from a remote URL via httpx."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from tcc_etl.extract.base import BaseExtractor


class HttpExtractor(BaseExtractor):
    """Extract data from an HTTP/HTTPS endpoint.

    Parameters
    ----------
    url:
        The URL to fetch. Must use the ``http`` or ``https`` scheme.
    timeout:
        Request timeout in seconds (default: 30).
    as_json:
        When ``True`` the response body is parsed as JSON and returned as a
        Python object.  When ``False`` the raw bytes are returned.

    Raises
    ------
    ValueError
        If *url* uses a scheme other than ``http`` or ``https``.
    """

    def __init__(self, url: str, *, timeout: float = 30.0, as_json: bool = True) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"Unsupported URL scheme '{parsed.scheme}'. Only http/https are allowed."
            )
        self.url = url
        self.timeout = timeout
        self.as_json = as_json

    async def extract(self) -> Any:  # noqa: ANN401
        """Fetch data from :attr:`url` asynchronously.

        Returns
        -------
        Any
            Parsed JSON object when ``as_json=True``, otherwise raw ``bytes``.

        Raises
        ------
        httpx.HTTPStatusError
            On 4xx / 5xx responses.
        httpx.TimeoutException
            When the request exceeds :attr:`timeout`.
        """
        logger.info("Extracting data from {}", self.url)
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url, timeout=self.timeout)
            response.raise_for_status()

        if self.as_json:
            data = response.json()
            logger.debug("Extracted {} records", len(data) if isinstance(data, list) else 1)
            return data

        raw: bytes = response.content
        logger.debug("Extracted {} bytes", len(raw))
        return raw
