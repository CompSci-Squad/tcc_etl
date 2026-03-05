"""ETL pipeline orchestrator.

Ties the extract → transform → load steps together and provides observability
(timing, logging, and basic error handling).
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from tcc_etl.extract.base import BaseExtractor
from tcc_etl.load.base import BaseLoader
from tcc_etl.transform.base import BaseTransformer


class Pipeline:
    """Orchestrate one full ETL run.

    Parameters
    ----------
    extractor:
        The configured extractor.
    transformer:
        The configured transformer.
    loader:
        The configured loader.
    destination_key:
        S3 object key (without prefix) where the result is stored.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        transformer: BaseTransformer,
        loader: BaseLoader,
        destination_key: str,
    ) -> None:
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        self.destination_key = destination_key

    # ------------------------------------------------------------------
    def run(self) -> Any:  # noqa: ANN401
        """Execute the pipeline (extract → transform → load).

        Returns
        -------
        Any
            The transformed data (useful for testing / chaining pipelines).

        Raises
        ------
        Exception
            Re-raises any exception after logging it.
        """
        logger.info("Pipeline starting - key={}", self.destination_key)
        start = time.perf_counter()

        try:
            raw = self._extract()
            transformed = self._transform(raw)
            self._load(transformed)
        except Exception as exc:
            logger.exception("Pipeline failed: {}", exc)
            raise

        elapsed = time.perf_counter() - start
        logger.info("Pipeline finished in {:.3f}s - key={}", elapsed, self.destination_key)
        return transformed

    # ------------------------------------------------------------------
    def _extract(self) -> Any:  # noqa: ANN401
        t0 = time.perf_counter()
        data = self.extractor.extract()
        logger.debug("Extract step completed in {:.3f}s", time.perf_counter() - t0)
        return data

    def _transform(self, data: Any) -> Any:  # noqa: ANN401
        t0 = time.perf_counter()
        result = self.transformer.transform(data)
        logger.debug("Transform step completed in {:.3f}s", time.perf_counter() - t0)
        return result

    def _load(self, data: Any) -> None:  # noqa: ANN401
        t0 = time.perf_counter()
        self.loader.load(data, self.destination_key)
        logger.debug("Load step completed in {:.3f}s", time.perf_counter() - t0)
