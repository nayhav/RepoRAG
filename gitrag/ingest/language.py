"""Language detection based on file extension."""

from __future__ import annotations

from pathlib import Path

from gitrag.core.types import Language

_EXTENSION_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".h": Language.C,
    ".c": Language.C,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".md": Language.MARKDOWN,
    ".rst": Language.RST,
    ".txt": Language.TEXT,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".json": Language.JSON,
    ".toml": Language.TOML,
}


def detect_language(path: Path) -> Language:
    """Return the Language for a file based on its extension."""
    return _EXTENSION_MAP.get(path.suffix.lower(), Language.UNKNOWN)
