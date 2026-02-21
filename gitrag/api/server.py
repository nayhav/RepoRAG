"""FastAPI server for GitRAG."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gitrag.config import load_config
from gitrag.core.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    repo_path: str
    force: bool = False


class IndexResponse(BaseModel):
    total_files: int
    total_chunks: int
    languages: list[str]
    duration_seconds: float


class QueryRequest(BaseModel):
    repo_path: str
    question: str
    conversation_id: str | None = None


class CitationResponse(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str = ""


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    conversation_id: str


class StatusResponse(BaseModel):
    repo_path: str
    index_path: str
    index_exists: bool
    chunks_count: int = 0
    bm25_exists: bool = False
    graph_exists: bool = False


class HealthResponse(BaseModel):
    status: str


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

_pipelines: dict[str, RAGPipeline] = {}
_config_path: str | None = None


def _get_pipeline(repo_path: str) -> RAGPipeline:
    """Return a cached pipeline for *repo_path*, creating one if needed."""
    if repo_path not in _pipelines:
        cfg = load_config(_config_path)
        _pipelines[repo_path] = RAGPipeline(repo_path, cfg)
    return _pipelines[repo_path]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config_path: str | None = None) -> FastAPI:
    """Create and return a FastAPI application."""
    global _config_path  # noqa: PLW0603
    _config_path = config_path

    app = FastAPI(title="GitRAG", version="0.1.0", description="RAG API for local Git repositories")

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.post("/index", response_model=IndexResponse)
    async def index_repo(req: IndexRequest) -> IndexResponse:
        """Index a local repository."""
        try:
            pipeline = _get_pipeline(req.repo_path)
            stats = pipeline.index(force=req.force)
        except Exception as exc:
            logger.exception("Indexing failed for %s", req.repo_path)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return IndexResponse(
            total_files=stats.total_files,
            total_chunks=stats.total_chunks,
            languages=stats.languages,
            duration_seconds=stats.duration_seconds,
        )

    @app.post("/query", response_model=QueryResponse)
    async def query_repo(req: QueryRequest) -> QueryResponse:
        """Query an indexed repository."""
        try:
            pipeline = _get_pipeline(req.repo_path)
            answer, conversation = pipeline.query(
                req.question,
                conversation_id=req.conversation_id,
            )
        except Exception as exc:
            logger.exception("Query failed for %s", req.repo_path)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        citations = [
            CitationResponse(
                file_path=c.file_path,
                start_line=c.start_line,
                end_line=c.end_line,
                symbol_name=c.symbol_name,
            )
            for c in answer.citations
        ]

        return QueryResponse(
            answer=answer.content,
            citations=citations,
            conversation_id=conversation.conversation_id,
        )

    @app.get("/status/{repo_path:path}", response_model=StatusResponse)
    async def repo_status(repo_path: str) -> StatusResponse:
        """Get index status for a repository."""
        try:
            pipeline = _get_pipeline(repo_path)
            info = pipeline.get_status()
        except Exception as exc:
            logger.exception("Status check failed for %s", repo_path)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return StatusResponse(
            repo_path=info["repo_path"],
            index_path=info["index_path"],
            index_exists=info["index_exists"],
            chunks_count=info.get("chunks_count", 0),
            bm25_exists=info.get("bm25_exists", False),
            graph_exists=info.get("graph_exists", False),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check."""
        return HealthResponse(status="ok")

    return app
