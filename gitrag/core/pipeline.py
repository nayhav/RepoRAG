"""High-level pipeline that wires together ingest → chunk → embed → index → retrieve → generate."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gitrag.chunking.ast_chunker import ASTChunker
from gitrag.config import GitRAGConfig, load_config
from gitrag.core.types import (
    CodeChunk,
    Conversation,
    GeneratedAnswer,
    ParsedQuery,
    QueryIntent,
    RetrievalResult,
)
from gitrag.embeddings.local import LocalEmbedder
from gitrag.generation.context import ContextCompressor
from gitrag.generation.llm import LLMClient
from gitrag.generation.prompts import build_context_prompt, build_system_prompt
from gitrag.index.bm25_store import BM25Store
from gitrag.index.graph_store import DependencyGraph
from gitrag.index.vector_store import VectorStore
from gitrag.ingest.loader import RepoLoader
from gitrag.memory.conversation import ConversationMemory
from gitrag.query.intent import IntentClassifier
from gitrag.query.reformulator import QueryReformulator
from gitrag.retrieval.hybrid import HybridRetriever
from gitrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index stats
# ---------------------------------------------------------------------------

@dataclass
class IndexStats:
    """Summary returned after indexing a repository."""

    total_files: int = 0
    total_chunks: int = 0
    languages: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end orchestrator for GitRAG.

    Holds all sub-components and exposes ``index``, ``query``, and ``chat``
    as the main public entry points.
    """

    def __init__(self, repo_path: str | Path, config: GitRAGConfig | None = None) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config or GitRAGConfig()

        # Persist directory is relative to the repo
        persist_dir = self.repo_path / self.config.index.persist_dir
        self.config.index.persist_dir = str(persist_dir)

        # Sub-components (lazy where possible)
        self._embedder = LocalEmbedder(self.config.embeddings)
        self._vector_store: VectorStore | None = None
        self._bm25_store: BM25Store | None = None
        self._graph: DependencyGraph | None = None
        self._retriever: HybridRetriever | None = None
        self._llm: LLMClient | None = None
        self._memory = ConversationMemory(self.config.memory)
        self._intent_classifier = IntentClassifier()
        self._reformulator = QueryReformulator()
        self._compressor = ContextCompressor(self.config.generation.max_context_tokens)

        # In-memory chunk store for graph expansion look-ups
        self._chunks: list[CodeChunk] = []
        self._chunks_by_file: dict[str, list[str]] = {}

        # Conversations keyed by id
        self._conversations: dict[str, Conversation] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, *, force: bool = False) -> IndexStats:
        """Ingest, chunk, embed, and index the repository.

        Parameters
        ----------
        force:
            When *True* the existing index is cleared before re-indexing.
        """
        start = time.time()

        # 1. Load files
        loader = RepoLoader(self.repo_path, self.config.ingest)
        files = loader.load_all()
        logger.info("Loaded %d files", len(files))

        # 2. Chunk
        chunker = ASTChunker(self.config.chunking)
        chunks: list[CodeChunk] = []
        for f in files:
            chunks.extend(chunker.chunk(f))
        logger.info("Produced %d chunks", len(chunks))

        # 3. Embed
        texts = [c.to_index_text() for c in chunks]
        embeddings = self._embedder.embed_documents(texts)
        logger.info("Embedded %d chunks", len(chunks))

        # 4. Index — vector store
        vs = self._get_vector_store()
        if force:
            vs.clear()
        vs.add_chunks(chunks, embeddings)

        # 5. Index — BM25
        bm25 = self._get_bm25_store()
        bm25.build(chunks)
        bm25.save(Path(self.config.index.persist_dir) / "bm25.pkl")

        # 6. Index — dependency graph
        graph = self._get_graph()
        all_file_paths = sorted({c.file_path for c in chunks})
        graph.build_from_chunks(chunks, all_file_paths)
        graph.save(Path(self.config.index.persist_dir) / "graph.json")

        # Cache chunks for later look-ups
        self._chunks = chunks
        self._chunks_by_file = {}
        for c in chunks:
            self._chunks_by_file.setdefault(c.file_path, []).append(c.chunk_id)

        languages = sorted({c.language.value for c in chunks})
        duration = time.time() - start

        return IndexStats(
            total_files=len(files),
            total_chunks=len(chunks),
            languages=languages,
            duration_seconds=round(duration, 2),
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        conversation_id: str | None = None,
    ) -> tuple[GeneratedAnswer, Conversation]:
        """Run the full RAG pipeline for a single question.

        Returns the generated answer and the conversation object.
        """
        self._ensure_index_loaded()

        # Conversation handling
        if conversation_id and conversation_id in self._conversations:
            conversation = self._conversations[conversation_id]
        else:
            conversation_id = conversation_id or uuid.uuid4().hex[:12]
            conversation = Conversation(
                conversation_id=conversation_id,
                repo_path=str(self.repo_path),
            )
            self._conversations[conversation_id] = conversation

        # 1. Intent classification
        intent = self._intent_classifier.classify(question)
        depth = IntentClassifier.get_retrieval_depth(intent)

        # 2. Query reformulation (for follow-ups)
        history_dicts = [
            {"role": t.role, "content": t.content} for t in conversation.turns
        ]
        reformulated = self._reformulator.reformulate(question, history_dicts)

        # 3. Retrieve
        retriever = self._get_retriever()
        results = retriever.retrieve(reformulated, top_k=depth)

        # 4. Compress context
        results = self._compressor.deduplicate(results)
        results = self._compressor.compress(results)

        # 5. Build prompts
        system_prompt = build_system_prompt()
        context_prompt = build_context_prompt(results, self.config.generation.max_context_tokens)
        conv_context = self._memory.get_context_for_llm(conversation)

        # 6. Generate
        llm = self._get_llm()
        answer = llm.generate(
            system_prompt=system_prompt,
            context=context_prompt,
            query=reformulated,
            conversation_history=conv_context,
            chunks=results,
        )

        # 7. Update conversation
        self._memory.add_turn(conversation, "user", question)
        cited_ids = [c.chunk_id for c in answer.citations]
        self._memory.add_turn(conversation, "assistant", answer.content, cited_ids)

        return answer, conversation

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return information about the current index."""
        persist = Path(self.config.index.persist_dir)
        exists = persist.exists()

        info: dict[str, Any] = {
            "repo_path": str(self.repo_path),
            "index_path": str(persist),
            "index_exists": exists,
        }

        if exists:
            try:
                vs = self._get_vector_store()
                info["chunks_count"] = vs.count()
            except Exception:
                info["chunks_count"] = 0

            bm25_path = persist / "bm25.pkl"
            info["bm25_exists"] = bm25_path.exists()

            graph_path = persist / "graph.json"
            info["graph_exists"] = graph_path.exists()

        return info

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def clear_conversation(self, conversation_id: str) -> None:
        conv = self._conversations.get(conversation_id)
        if conv:
            self._memory.clear(conv)

    # ------------------------------------------------------------------
    # Lazy initialisers
    # ------------------------------------------------------------------

    def _get_vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(self.config.index)
        return self._vector_store

    def _get_bm25_store(self) -> BM25Store:
        if self._bm25_store is None:
            self._bm25_store = BM25Store()
            bm25_path = Path(self.config.index.persist_dir) / "bm25.pkl"
            if bm25_path.exists():
                self._bm25_store.load(bm25_path)
        return self._bm25_store

    def _get_graph(self) -> DependencyGraph:
        if self._graph is None:
            self._graph = DependencyGraph()
            graph_path = Path(self.config.index.persist_dir) / "graph.json"
            if graph_path.exists():
                self._graph.load(graph_path)
        return self._graph

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            reranker: Reranker | None = None
            if self.config.retrieval.enable_reranking:
                reranker = Reranker(self.config.retrieval.reranker_model)
            self._retriever = HybridRetriever(
                config=self.config.retrieval,
                vector_store=self._get_vector_store(),
                bm25_store=self._get_bm25_store(),
                embedder=self._embedder,
                reranker=reranker,
            )
        return self._retriever

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self.config.generation)
        return self._llm

    def _ensure_index_loaded(self) -> None:
        """Make sure index stores are initialised (loads from disk if persisted)."""
        self._get_vector_store()
        self._get_bm25_store()
        self._get_graph()
