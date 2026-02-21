"""File ingestion: discovery, filtering, and language detection."""

from .filters import FileFilter
from .language import detect_language
from .loader import RepoLoader

__all__ = ["FileFilter", "RepoLoader", "detect_language"]
