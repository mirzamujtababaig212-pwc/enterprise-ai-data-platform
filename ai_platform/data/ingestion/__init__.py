"""Data ingestion implementations."""

from ai_platform.data.ingestion.base import BaseDataReader
from ai_platform.data.ingestion.local_file import LocalFileReader

__all__ = [
    "BaseDataReader",
    "LocalFileReader",
]
