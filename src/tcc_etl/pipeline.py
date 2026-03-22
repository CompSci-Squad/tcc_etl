"""ETL pipeline orchestrator.

Ties the extract → transform → load steps together and provides observability
(timing, logging, and basic error handling).

The pipeline is fully async. ``run()`` should be called with
``asyncio.run(pipeline.run())`` or ``await pipeline.run()`` inside an async
context.
"""

from __future__ import annotations

import time
from pathlib import Path
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
        S3 object key template (without prefix). Multi-batch runs produce
        numbered parts, e.g. ``output.parquet`` →
        ``output_part0000.parquet``, ``output_part0001.parquet``, ...
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
    @staticmethod
    def _batch_key(stem: str, suffix: str, index: int) -> str:
        """Build a partitioned object key.

        Example: stem="output", suffix=".parquet", index=3
        → ``output_part0003.parquet``
        """
        return f"{stem}_part{index:04d}{suffix}"

    # ------------------------------------------------------------------
    async def run(self) -> int:
        """Execute the pipeline (extract → transform → load).

        Returns
        -------
        int
            The number of batches that were uploaded.

        Raises
        ------
        Exception
            Re-raises any exception after logging it.
        """
        logger.info("Pipeline starting - key={}", self.destination_key)
        start = time.perf_counter()

        p = Path(self.destination_key)
        stem = p.stem
        suffix = p.suffix
        batch_count = 0

        try:
            raw = await self._extract()
            for batch in self._transform(raw):
                key = self._batch_key(stem, suffix, batch_count)
                await self._load(batch, key)
                batch_count += 1
        except Exception as exc:
            logger.exception("Pipeline failed: {}", exc)
            raise

        elapsed = time.perf_counter() - start
        logger.info(
            "Pipeline finished in {:.3f}s - {} batch(es) - key={}",
            elapsed,
            batch_count,
            self.destination_key,
        )
        return batch_count

    # ------------------------------------------------------------------
    async def _extract(self) -> Any:  # noqa: ANN401
        t0 = time.perf_counter()
        data = await self.extractor.extract()
        logger.debug("Extract step completed in {:.3f}s", time.perf_counter() - t0)
        return data

    def _transform(self, data: Any) -> Any:  # noqa: ANN401
        t0 = time.perf_counter()
        result = self.transformer.transform(data)
        logger.debug("Transform step initialised in {:.3f}s", time.perf_counter() - t0)
        return result

    async def _load(self, data: Any, key: str) -> None:  # noqa: ANN401
        t0 = time.perf_counter()
        await self.loader.load(data, key)
        logger.debug("Load step completed in {:.3f}s - key={}", time.perf_counter() - t0, key)
