"""Configuration management for GitRAG.

Loads configuration from YAML with sensible defaults.
Supports override via environment variables prefixed with GITRAG_.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Typed config sections
# ---------------------------------------------------------------------------

@dataclass
class IngestConfig:
    ignore_dirs: list[str] = field(default_factory=lambda: [
        ".git", "node_modules", "vendor", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache", "target",
        ".next", ".nuxt",
    ])
    ignore_patterns: list[str] = field(default_factory=lambda: [
        "*.pyc", "*.so", "*.dylib", "*.dll", "*.exe", "*.o", "*.a",
        "*.bin", "*.dat", "*.db", "*.sqlite",
        "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.ico", "*.svg",
        "*.woff", "*.woff2", "*.ttf", "*.eot",
        "*.mp3", "*.mp4", "*.avi", "*.mov",
        "*.zip", "*.tar", "*.gz", "*.rar", "*.7z",
        "*.lock", "package-lock.json", "yarn.lock",
        "*.min.js", "*.min.css", "*.map",
    ])
    max_file_size_kb: int = 512


@dataclass
class ChunkingConfig:
    max_chunk_tokens: int = 400
    min_chunk_tokens: int = 30
    context_lines: int = 5
    include_docstrings: bool = True
    include_imports: bool = True


@dataclass
class EmbeddingsConfig:
    model_name: str = "BAAI/bge-base-en-v1.5"
    dimensions: int = 768
    batch_size: int = 64
    device: str = ""
    normalize: bool = True
    query_prefix: str = "Represent this sentence for searching relevant code: "


@dataclass
class IndexConfig:
    vector_backend: str = "chroma"
    persist_dir: str = ".gitrag_index"
    collection_name: str = "gitrag"


@dataclass
class RetrievalConfig:
    vector_top_k: int = 50
    bm25_top_k: int = 50
    fusion_top_k: int = 30
    rerank_top_k: int = 10
    rrf_k: int = 60
    reranker_model: str = "BAAI/bge-reranker-base"
    enable_reranking: bool = True


@dataclass
class GraphConfig:
    max_hops: int = 2
    max_expand_per_hop: int = 15
    file_level: bool = True
    symbol_level: bool = False


@dataclass
class QueryConfig:
    intent_classification: bool = True
    reformulation: bool = True


@dataclass
class MemoryConfig:
    max_turns: int = 20
    augmented_memory: bool = True
    summary_interval: int = 5


@dataclass
class GenerationConfig:
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    context_window: int = 8192
    context_compression: bool = True
    max_context_tokens: int = 6000


@dataclass
class EvaluationConfig:
    enabled: bool = False
    output_dir: str = ".gitrag_eval"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class GitRAGConfig:
    ingest: IngestConfig = field(default_factory=IngestConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _apply_env_overrides(cfg: GitRAGConfig) -> None:
    """Override config values with GITRAG_ environment variables.

    Convention: GITRAG_<SECTION>_<KEY> e.g. GITRAG_GENERATION_MODEL=codellama
    """
    prefix = "GITRAG_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section_name, field_name = parts
        section = getattr(cfg, section_name, None)
        if section is None:
            continue
        if not hasattr(section, field_name):
            continue
        current = getattr(section, field_name)
        # Coerce type
        if isinstance(current, bool):
            setattr(section, field_name, value.lower() in ("1", "true", "yes"))
        elif isinstance(current, int):
            setattr(section, field_name, int(value))
        elif isinstance(current, float):
            setattr(section, field_name, float(value))
        else:
            setattr(section, field_name, value)


def _dict_to_dataclass(dc_cls: type, data: dict[str, Any]) -> Any:
    """Recursively populate a dataclass from a dict, ignoring unknown keys."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(dc_cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return dc_cls(**filtered)


def load_config(path: Path | str | None = None) -> GitRAGConfig:
    """Load configuration from a YAML file with defaults and env overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                raw = yaml.safe_load(f) or {}

    cfg = GitRAGConfig(
        ingest=_dict_to_dataclass(IngestConfig, raw.get("ingest", {})),
        chunking=_dict_to_dataclass(ChunkingConfig, raw.get("chunking", {})),
        embeddings=_dict_to_dataclass(EmbeddingsConfig, raw.get("embeddings", {})),
        index=_dict_to_dataclass(IndexConfig, raw.get("index", {})),
        retrieval=_dict_to_dataclass(RetrievalConfig, raw.get("retrieval", {})),
        graph=_dict_to_dataclass(GraphConfig, raw.get("graph", {})),
        query=_dict_to_dataclass(QueryConfig, raw.get("query", {})),
        memory=_dict_to_dataclass(MemoryConfig, raw.get("memory", {})),
        generation=_dict_to_dataclass(GenerationConfig, raw.get("generation", {})),
        evaluation=_dict_to_dataclass(EvaluationConfig, raw.get("evaluation", {})),
    )
    _apply_env_overrides(cfg)
    return cfg
