# RepoRAG

**Production-grade local RAG system for chatting with any GitHub repository.**

RepoRAG ingests a local repository, builds a semantic + structural index using AST-aware chunking and hybrid search, and supports conversational multi-turn querying — all fully offline.

---

## Architecture

### Design Decision


**Selected Architecture: B) Microkernel + Plugin.** Each component (parser, embedder, retriever, generator) implements an abstract interface and is swappable without re-plumbing the pipeline. This avoids the rigidity of a monolith and the over-engineering of a DAG for a local tool.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          RepoRAG Pipeline                            │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌───────────┐ │
│  │  Ingest   │──▶│  AST Chunker │──▶│  Embedder  │──▶│   Index   │ │
│  │  (Loader) │   │ (tree-sitter)│   │  (BGE/etc) │   │ (Vector + │ │
│  └──────────┘   └──────────────┘   └────────────┘   │  BM25 +   │ │
│                                                      │  Graph)   │ │
│                                                      └─────┬─────┘ │
│                                                            │       │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐         │       │
│  │  Query    │──▶│   Hybrid     │──▶│  Reranker  │◀────────┘       │
│  │ Underst.  │   │  Retriever   │   │ (CrossEnc) │                 │
│  └──────────┘   └──────────────┘   └──────┬─────┘                 │
│       │                                    │                       │
│       │              ┌─────────────────────▼─────────────────┐     │
│       └──────────────▶│         LLM Generator               │     │
│                       │  (Ollama / OpenAI-compatible)        │     │
│  ┌──────────┐        │  + Context Compression               │     │
│  │ Memory   │◀──────▶│  + Citation Extraction               │     │
│  └──────────┘        └───────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Repository Files
    │
    ▼
File Loader (language detection, binary filtering)
    │
    ▼
AST Chunker (tree-sitter) ──── Fallback: Text Chunker
    │
    ├──▶ Vector Store (ChromaDB, cosine similarity)
    ├──▶ BM25 Store (rank_bm25, code-aware tokenization)
    └──▶ Dependency Graph (networkx, import resolution)
    
User Query
    │
    ▼
Intent Classification ──▶ Retrieval Depth Adjustment
    │
    ▼
Query Reformulation (for follow-ups)
    │
    ▼
Hybrid Retrieval
    ├── Dense: Vector similarity search
    ├── Sparse: BM25 keyword search
    └── Reciprocal Rank Fusion (RRF)
    │
    ▼
Cross-Encoder Reranking
    │
    ▼
Multi-Hop Graph Expansion (optional)
    │
    ▼
Context Compression + Deduplication
    │
    ▼
LLM Generation (with citations)
    │
    ▼
Answer + [file:line] Citations
```

---

## Why AST-Aware Chunking?

Naive fixed-size chunking (e.g., 512-token windows with overlap) is fundamentally broken for code:

1. **Semantic boundary violation**: A 512-token window can split a function in half, losing the connection between signature and body.
2. **Context loss**: The chunk loses import context, class membership, and docstrings.
3. **Redundant overlap**: Sliding-window overlap wastes embedding space on duplicated content.
4. **No structural metadata**: Fixed chunks can't carry symbol names, file paths, or dependency references.
5. **Cross-language inconsistency**: Code structure varies by language; fixed windows ignore this.

AST-aware chunking extracts **function, class, and method boundaries** directly from the parse tree, preserving:
- Complete function/method bodies as atomic units
- Docstrings attached to their parent symbols
- Import context for dependency resolution
- Structural metadata (symbol kind, parent class, line numbers)

---

## Why Hybrid Search > Pure Vector?

Pure vector search fails on:
- **Exact symbol names**: "Where is `calculateTaxRate` defined?" — vector search may miss the exact token match.
- **Rare identifiers**: Unusual function names have poor embedding coverage.
- **Boolean precision**: "Find all files importing `redis`" needs keyword matching.

BM25 excels at exact matches; vector search excels at semantic similarity. **Reciprocal Rank Fusion (RRF)** combines them:

```
score(d) = Σ 1/(k + rank_i(d))  for each retriever i
```

This is rank-based (not score-based), making it robust to different score distributions.

---

## Technical Decisions

| Decision | Choice | Justification |
|---|---|---|
| AST Parser | tree-sitter | Uniform multi-language support, fast C bindings, consistent node metadata |
| Embedding Model | BAAI/bge-base-en-v1.5 | 768d, strong general performance, sentence-transformers compatible |
| Vector DB | ChromaDB | Local-first, built-in persistence, metadata filtering, adequate for repo-scale |
| BM25 | rank_bm25 | Simple, reliable, code-aware tokenization (camelCase/snake_case splitting) |
| Reranker | BAAI/bge-reranker-base | Cross-encoder quality boost, bounded candidate set (≤30) |
| Similarity | Cosine (normalized) | Standard for text embeddings, L2-normalized in store |
| LLM | Ollama (local) | Fully offline, OpenAI-compatible API, any local model |
| Graph | networkx DiGraph | Lightweight, file-level import resolution, BFS traversal |

---

## Installation

```bash
# Clone and install
git clone <repo-url> && cd gitrag
pip install -e .

# Or with Docker
docker build -t gitrag .
```

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally (for LLM generation)

```bash
# Pull an LLM model
ollama pull llama3.1:8b
```

---

## Usage

### CLI

```bash
# Index a repository
gitrag index /path/to/your/repo

# Single-shot query
gitrag query /path/to/your/repo "How does authentication work?"

# Interactive chat
gitrag chat /path/to/your/repo

# Check index status
gitrag status /path/to/your/repo
```

### Chat Commands

Inside `gitrag chat`:
- `/quit` — Exit
- `/clear` — Clear conversation history
- `/stats` — Show index statistics

### Custom Config

```bash
gitrag index /path/to/repo --config my_config.yaml
```

### API Server

```bash
# Start the API
uvicorn gitrag.api.server:create_app --factory --host 0.0.0.0 --port 8000

# Or with Docker
docker run -p 8000:8000 -v /path/to/repos:/repos gitrag
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| POST | `/index` | Index a repository |
| POST | `/query` | Query with conversation support |
| GET | `/status/{repo_path}` | Index status |
| GET | `/health` | Health check |

```bash
# Example: Index
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/repos/my-project"}'

# Example: Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/repos/my-project", "question": "What does the main function do?"}'
```

---

## Configuration

All settings in `config.yaml`. Key sections:

```yaml
embeddings:
  model_name: "BAAI/bge-base-en-v1.5"  # Local embedding model
  device: ""                            # auto-detect (cuda/mps/cpu)

generation:
  provider: "ollama"                    # or "openai_compatible"
  model: "llama3.1:8b"                 # any Ollama model
  base_url: "http://localhost:11434"
  temperature: 0.1                     # low for deterministic answers

retrieval:
  enable_reranking: true               # disable for faster but lower quality
  rerank_top_k: 10                     # final results count
```

Environment variable overrides: `GITRAG_<SECTION>_<KEY>`, e.g.:
```bash
export GITRAG_GENERATION_MODEL=codellama:13b
export GITRAG_EMBEDDINGS_DEVICE=cuda
```

---

## Project Structure

```
gitrag/
├── core/
│   ├── types.py          # All data types (CodeChunk, RetrievalResult, etc.)
│   └── pipeline.py       # Main orchestrator
├── ingest/
│   ├── loader.py         # Repository file discovery
│   ├── language.py       # Language detection
│   └── filters.py        # Binary/ignore filtering
├── chunking/
│   ├── ast_chunker.py    # tree-sitter AST-aware chunking
│   └── text_chunker.py   # Fallback for docs/config
├── embeddings/
│   └── local.py          # Sentence-transformers embedder
├── index/
│   ├── vector_store.py   # ChromaDB vector index
│   ├── bm25_store.py     # BM25 sparse index
│   └── graph_store.py    # Dependency graph (networkx)
├── retrieval/
│   ├── hybrid.py         # Hybrid retriever orchestrator
│   ├── fusion.py         # Reciprocal Rank Fusion
│   └── reranker.py       # Cross-encoder reranking
├── query/
│   ├── intent.py         # Intent classification
│   ├── reformulator.py   # Follow-up query reformulation
│   └── multi_hop.py      # Graph-based context expansion
├── memory/
│   └── conversation.py   # Multi-turn conversation state
├── generation/
│   ├── llm.py            # LLM client (Ollama/OpenAI)
│   ├── prompts.py        # Intent-specific prompt templates
│   └── context.py        # Context compression
├── evaluation/
│   └── metrics.py        # Precision@k, MRR, NDCG, faithfulness
├── api/
│   └── server.py         # FastAPI REST API
└── cli.py                # Click CLI
```

---

## Evaluation Framework

Built-in metrics for measuring retrieval and generation quality:

**Retrieval Metrics:**
- Precision@k, Recall@k
- MRR (Mean Reciprocal Rank)
- NDCG@k (Normalized Discounted Cumulative Gain)

**Generation Metrics:**
- Faithfulness score (token-overlap with context)
- Citation coverage (paragraphs with citations)
- Hallucination score (1 - faithfulness + code-block penalty)

**Benchmarking:**
- Synthetic query generation from indexed chunks
- Latency measurement per pipeline stage

---

## Hallucination Mitigation

1. **Low temperature** (0.1): Reduces creative/hallucinated responses.
2. **Strict system prompt**: "You MUST base answers strictly on provided context."
3. **Citation requirement**: Every claim must reference `[file:line]`.
4. **Context grounding**: Only retrieved, verified code chunks are in the prompt.
5. **Faithfulness scoring**: Post-hoc validation that answer tokens appear in context.

---

## Memory Strategy

Three-tier conversation memory:

1. **Short-term buffer**: Last N turns kept verbatim (default: 20).
2. **Rolling summary**: Older turns compressed into extractive summary every 5 turns.
3. **Context window optimization**: Summary + recent turns fit within token budget; oldest dropped first.

Query reformulation detects follow-up queries (pronouns, short queries) and prepends context from prior turns to create standalone queries.

---

## Future Improvements

- [ ] Incremental indexing (only re-index changed files via content hash)
- [ ] Code-optimized embedding model (jina-embeddings-v2-base-code)
- [ ] Symbol-level call graph (not just file-level imports)
- [ ] Streaming LLM responses
- [ ] Web UI (React frontend)
- [ ] Multi-repo support (query across repositories)
- [ ] Git blame integration (who changed what)
- [ ] Learned fusion weights (instead of fixed RRF)
- [ ] FAISS backend option for larger repos
- [ ] IDE plugins (VS Code extension)

---

## License

MIT
