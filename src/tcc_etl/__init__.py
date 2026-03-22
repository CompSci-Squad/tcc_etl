"""tcc_etl - high-performance ETL pipeline (extract → transform → load → S3)."""

from importlib.metadata import version

__version__ = version("tcc-etl")

__all__ = ["__version__"]
