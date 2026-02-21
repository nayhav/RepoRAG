"""Main repository file loader."""

from __future__ import annotations

import logging
from pathlib import Path

from gitrag.config import IngestConfig
from gitrag.core.types import IngestedFile

from .filters import FileFilter
from .language import detect_language

log = logging.getLogger(__name__)


class RepoLoader:
    """Walk a local repository and load files as IngestedFile objects."""

    def __init__(self, repo_path: Path, config: IngestConfig | None = None) -> None:
        self._root = repo_path.resolve()
        self._config = config or IngestConfig()
        self._filter = FileFilter(self._config)

    def load_all(self) -> list[IngestedFile]:
        """Discover and load all eligible files from the repository."""
        files: list[IngestedFile] = []
        dirs_to_walk: list[Path] = [self._root]

        while dirs_to_walk:
            current = dirs_to_walk.pop()
            try:
                entries = sorted(current.iterdir())
            except OSError:
                log.warning("Cannot read directory: %s", current)
                continue

            for entry in entries:
                if entry.is_dir():
                    if not self._filter.should_skip_dir(entry.name):
                        dirs_to_walk.append(entry)
                    continue

                if not entry.is_file():
                    continue

                if self._filter.should_skip_file(entry):
                    continue

                ingested = self._load_file(entry)
                if ingested is not None:
                    files.append(ingested)

        log.info("Loaded %d files from %s", len(files), self._root)
        return files

    def _load_file(self, path: Path) -> IngestedFile | None:
        """Read a single file and return an IngestedFile, or None on failure."""
        content, encoding = _read_text(path)
        if content is None:
            log.debug("Skipping unreadable file: %s", path)
            return None

        return IngestedFile(
            path=path.relative_to(self._root),
            abs_path=path,
            content=content,
            language=detect_language(path),
            size_bytes=path.stat().st_size,
            encoding=encoding,
        )


def _read_text(path: Path) -> tuple[str | None, str]:
    """Read file content, trying utf-8 first then latin-1 as fallback."""
    try:
        return path.read_text(encoding="utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    except OSError:
        return None, "utf-8"

    try:
        return path.read_text(encoding="latin-1"), "latin-1"
    except OSError:
        return None, "latin-1"
