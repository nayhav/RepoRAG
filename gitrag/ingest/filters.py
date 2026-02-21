"""File filtering logic for repository ingestion."""

from __future__ import annotations

import fnmatch
from pathlib import Path

from gitrag.config import IngestConfig


class FileFilter:
    """Decides whether a file or directory should be skipped during ingestion."""

    def __init__(self, config: IngestConfig) -> None:
        self._ignore_dirs = set(config.ignore_dirs)
        self._ignore_patterns = config.ignore_patterns
        self._max_size = config.max_file_size_kb * 1024

    def should_skip_dir(self, name: str) -> bool:
        """Return True if a directory should not be recursed into."""
        return name in self._ignore_dirs

    def should_skip_file(self, path: Path) -> bool:
        """Return True if a file should be excluded from ingestion."""
        name = path.name
        for pattern in self._ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True

        try:
            size = path.stat().st_size
        except OSError:
            return True

        if size > self._max_size:
            return True

        return _is_binary(path)


def _is_binary(path: Path) -> bool:
    """Detect binary files by checking for null bytes in the first 8 KB."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True
