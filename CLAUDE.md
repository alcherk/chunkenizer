# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chunkenizer is a Python webservice for RAG (Retrieval-Augmented Generation) document ingestion and semantic search. It provides document upload, intelligent chunking, embedding generation, and semantic search capabilities.

## Architecture

```
┌─────────────┐
│   FastAPI   │
│   (API+UI)  │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
┌──▼──┐ ┌──▼────┐
│SQLite│ │Qdrant │
│(Meta)│ │(Vecs) │
└──────┘ └───────┘
```

**Dual-Database Design:**
- **SQLite** (`data/chunkenizer.db`): Document metadata only (names, hashes, chunk/token counts, timestamps)
- **Qdrant**: Vector embeddings (384-dimensional from `all-MiniLM-L6-v2`)

**Key Modules:**
- `app/api/routes.py` - REST API endpoints (document CRUD, search)
- `app/ui/routes.py` - Web UI routes with Jinja2 templates
- `app/ingest/chunker.py` - Token-based text splitting with tiktoken
- `app/ingest/processor.py` - Document processing pipeline (text extraction, hashing)
- `app/vectorstore/qdrant_client.py` - Qdrant vector DB wrapper
- `app/config.py` - Pydantic-settings configuration from environment

## Commands

```bash
# Local development
python -m app.main                    # Run application
uvicorn app.main:app --reload         # Run with auto-reload

# Testing
pytest tests/                         # Run all tests
pytest tests/test_chunker.py -v       # Run specific test file

# Docker
make up                               # Start services (docker-compose up -d)
make down                             # Stop services
make logs                             # View logs
make build                            # Build Docker images

# Database inspection
make inspect-db                       # Inspect SQLite database
make inspect-qdrant                   # Inspect Qdrant vectors
python scripts/inspect_db.py          # Interactive SQLite inspector
python scripts/inspect_qdrant.py      # Qdrant vector inspector
```

## Key Patterns

**Lazy Initialization** (`app/api/routes.py`): Global `_processor` and `_vectorstore` are initialized on first use via `get_processor()` and `get_vectorstore()` functions.

**SHA256 Deduplication**: Documents are hashed to prevent duplicates. Check `existing_doc = db.query(Document).filter(Document.sha256 == sha256).first()` before ingestion.

**Token-Based Chunking** (`app/ingest/chunker.py`): Uses tiktoken for precise token counting. Configurable via `CHUNK_SIZE_TOKENS` (default 500) and `CHUNK_OVERLAP_TOKENS` (default 50).

**Content-Type Processing** (`app/ingest/processor.py`): Handles markdown, plain text, and JSON with format-specific extraction logic.

## Configuration

Environment variables (see `.env.example`):
- `EMBEDDING_MODEL` - Sentence-transformers model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` - Chunking parameters
- `QDRANT_HOST` / `QDRANT_PORT` - Vector DB connection
- `SQLITE_PATH` - Metadata database path

## API Endpoints

- `POST /documents` - Upload document
- `GET /documents` - List documents
- `DELETE /documents/{doc_id}` - Delete document
- `POST /search` - Semantic search (query, top_k, filters)
- `GET /api/health` - Health check (includes Qdrant/SQLite status)
- `GET /docs` - Swagger UI

## Testing

Tests use pytest with async support. Test files:
- `tests/test_chunker.py` - Token chunking verification
- `tests/test_processor.py` - Text extraction and JSON flattening
- `tests/test_api.py` - API endpoint tests
