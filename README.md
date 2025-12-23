# Chunkenizer - RAG Ingestion & Retrieval System

A production-ready Python webservice for RAG (Retrieval-Augmented Generation) document ingestion and semantic search. Built with FastAPI, Qdrant vector database, and sentence-transformers for local embeddings.

## Features

- **Document Upload**: Support for markdown (.md), text (.txt), and JSON (.json) files via Web UI or API
- **Intelligent Chunking**: Token-based text splitting with configurable overlap using tiktoken
- **Local Embeddings**: Uses sentence-transformers for generating embeddings without external API calls
- **Vector Search**: Semantic search over document chunks using Qdrant
- **Web UI**: Clean, server-rendered interface for document management and search
- **Docker Deployment**: Ready-to-deploy with docker-compose including persistent storage

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
│SQLite│ │Qdrant│
│ (Meta)│ │(Vecs)│
└──────┘ └──────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Using Docker Compose (Recommended)

1. **Clone and navigate to the project**:
   ```bash
   cd Chunkenizer
   ```

2. **Start the services**:
   ```bash
   docker-compose up -d
   ```

3. **Access the application**:
   - Web UI: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/health

4. **Stop the services**:
   ```bash
   docker-compose down
   ```

### Local Development

1. **Create and activate a virtual environment**:
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional, defaults work):
   ```bash
   cp .env.example .env
   # Edit .env if needed
   ```

4. **Start Qdrant** (required):
   ```bash
   docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
   ```

5. **Update .env** for local Qdrant:
   ```env
   QDRANT_HOST=localhost
   ```

6. **Run the application**:
   ```bash
   python -m app.main
   # Or: uvicorn app.main:app --reload
   ```

7. **Deactivate virtual environment** (when done):
   ```bash
   deactivate
   ```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "status": "ok",
  "sqlite": "ok",
  "qdrant": "ok",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### Upload Document
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@document.txt" \
  -F "metadata_json={\"source\": \"example\"}" \
  -F "force_reindex=false"
```

Response:
```json
{
  "message": "Document ingested successfully",
  "document_id": "uuid-here",
  "chunk_count": 15,
  "total_tokens": 7500,
  "sha256": "hash-here"
}
```

### Search Documents
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5
  }'
```

Response:
```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "score": 0.85,
      "document_id": "uuid",
      "document_name": "document.txt",
      "chunk_index": 3,
      "chunk_text": "...",
      "token_count": 450,
      "metadata": {}
    }
  ],
  "total_results": 1
}
```

### List Documents
```bash
curl http://localhost:8000/documents
```

### Get Document Details
```bash
curl http://localhost:8000/documents/{doc_id}
```

### Delete Document
```bash
curl -X DELETE http://localhost:8000/documents/{doc_id}
```

## API Documentation for External Developers

### Base URL

- **Local Development**: `http://localhost:8000`
- **Production**: Replace with your deployment URL

### Authentication

Currently, the API does not require authentication. For production deployments, consider adding API keys or OAuth2.

### API Endpoints Reference

#### 1. Health Check

Check the health status of the service and its dependencies.

**Endpoint**: `GET /api/health`

**Request**:
```bash
curl http://localhost:8000/api/health
```

**Response** (200 OK):
```json
{
  "status": "ok",
  "sqlite": "ok",
  "qdrant": "ok",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Response Fields**:
- `status`: Overall service status (`"ok"` or `"error"`)
- `sqlite`: SQLite database status (`"ok"` or `"error"`)
- `qdrant`: Qdrant vector database status (`"ok"` or `"error"`)
- `embedding_model`: Name of the embedding model in use

---

#### 2. Upload Document

Upload and ingest a document. The document will be chunked, embedded, and stored in the vector database.

**Endpoint**: `POST /documents`

**Content-Type**: `multipart/form-data`

**Request Parameters**:
- `file` (required): File to upload (`.md`, `.txt`, or `.json`)
- `metadata_json` (optional): JSON string with custom metadata
- `force_reindex` (optional): Boolean, force reindexing if document already exists (default: `false`)

**Request Example**:
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@document.txt" \
  -F "metadata_json={\"source\": \"example\", \"category\": \"docs\"}" \
  -F "force_reindex=false"
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/documents"
files = {"file": open("document.txt", "rb")}
data = {
    "metadata_json": '{"source": "example", "category": "docs"}',
    "force_reindex": "false"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**JavaScript/TypeScript Example**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('metadata_json', JSON.stringify({source: "example", category: "docs"}));
formData.append('force_reindex', 'false');

fetch('http://localhost:8000/documents', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**Response** (200 OK):
```json
{
  "message": "Document ingested successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_count": 15,
  "total_tokens": 7500,
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

**Response** (200 OK - Document Already Exists):
```json
{
  "message": "Document already exists",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "chunk_count": 15,
  "total_tokens": 7500
}
```

**Error Response** (500 Internal Server Error):
```json
{
  "detail": "Error message describing what went wrong"
}
```

**Response Fields**:
- `message`: Status message
- `document_id`: UUID of the ingested document
- `chunk_count`: Number of chunks created
- `total_tokens`: Total token count across all chunks
- `sha256`: SHA256 hash of the document content

---

#### 3. Search Documents

Perform semantic search over ingested document chunks.

**Endpoint**: `POST /search`

**Content-Type**: `application/json`

**Request Body**:
```json
{
  "query": "string (required)",
  "top_k": 5,
  "filters": {
    "doc_id": "optional-document-id",
    "name": "optional-document-name"
  }
}
```

**Request Example**:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "filters": {}
  }'
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/search"
payload = {
    "query": "What is machine learning?",
    "top_k": 5,
    "filters": {}
}

response = requests.post(url, json=payload)
results = response.json()

for result in results["results"]:
    print(f"Score: {result['score']:.4f}")
    print(f"Document: {result['document_name']}")
    print(f"Chunk: {result['chunk_text'][:200]}...")
    print()
```

**JavaScript/TypeScript Example**:
```javascript
fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'What is machine learning?',
    top_k: 5,
    filters: {}
  })
})
.then(response => response.json())
.then(data => {
  data.results.forEach(result => {
    console.log(`Score: ${result.score}`);
    console.log(`Document: ${result.document_name}`);
    console.log(`Chunk: ${result.chunk_text.substring(0, 200)}...`);
  });
});
```

**Response** (200 OK):
```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "score": 0.8542,
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "document_name": "document.txt",
      "chunk_index": 3,
      "chunk_text": "Machine learning is a subset of artificial intelligence...",
      "token_count": 450,
      "metadata": {
        "source": "example",
        "category": "docs"
      }
    }
  ],
  "total_results": 1
}
```

**Response Fields**:
- `query`: The search query used
- `results`: Array of search results, sorted by relevance (highest score first)
  - `score`: Similarity score (0-1, higher is more similar)
  - `document_id`: UUID of the source document
  - `document_name`: Name of the source document
  - `chunk_index`: Index of the chunk within the document
  - `chunk_text`: Full text of the matching chunk
  - `token_count`: Number of tokens in the chunk
  - `metadata`: Custom metadata associated with the document
- `total_results`: Total number of results returned

---

#### 4. List Documents

Retrieve a list of all ingested documents with their metadata.

**Endpoint**: `GET /documents`

**Request Example**:
```bash
curl http://localhost:8000/documents
```

**Python Example**:
```python
import requests

response = requests.get("http://localhost:8000/documents")
documents = response.json()

for doc in documents["documents"]:
    print(f"{doc['name']}: {doc['chunk_count']} chunks, {doc['total_tokens']} tokens")
```

**Response** (200 OK):
```json
{
  "documents": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "document.txt",
      "content_type": "text/plain",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "created_at": "2024-01-15T10:30:00",
      "updated_at": "2024-01-15T10:30:00",
      "status": "ingested",
      "metadata_json": "{\"source\": \"example\"}",
      "chunk_count": 15,
      "total_tokens": 7500
    }
  ],
  "total": 1
}
```

---

#### 5. Get Document Details

Retrieve detailed information about a specific document.

**Endpoint**: `GET /documents/{doc_id}`

**Path Parameters**:
- `doc_id` (required): UUID of the document

**Request Example**:
```bash
curl http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000
```

**Response** (200 OK):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "document.txt",
  "content_type": "text/plain",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:30:00",
  "status": "ingested",
  "metadata_json": "{\"source\": \"example\"}",
  "chunk_count": 15,
  "total_tokens": 7500
}
```

**Error Response** (404 Not Found):
```json
{
  "detail": "Document not found"
}
```

---

#### 6. Delete Document

Delete a document and all its associated chunks from both the metadata database and vector store.

**Endpoint**: `DELETE /documents/{doc_id}`

**Path Parameters**:
- `doc_id` (required): UUID of the document

**Request Example**:
```bash
curl -X DELETE http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000
```

**Python Example**:
```python
import requests

doc_id = "550e8400-e29b-41d4-a716-446655440000"
response = requests.delete(f"http://localhost:8000/documents/{doc_id}")
print(response.json())
```

**Response** (200 OK):
```json
{
  "message": "Document deleted successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Response** (404 Not Found):
```json
{
  "detail": "Document not found"
}
```

---

### Error Handling

The API uses standard HTTP status codes:

- **200 OK**: Request succeeded
- **400 Bad Request**: Invalid request parameters
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server error

Error responses follow this format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Rate Limiting

Currently, there is no rate limiting implemented. For production deployments, consider implementing rate limiting to prevent abuse.

### Best Practices

1. **Health Checks**: Monitor `/api/health` regularly to ensure service availability
2. **Error Handling**: Always check HTTP status codes and handle errors gracefully
3. **Metadata**: Use `metadata_json` to store custom metadata for filtering and organization
4. **Batch Uploads**: Upload multiple documents sequentially rather than in parallel to avoid overwhelming the service
5. **Search Optimization**: 
   - Use specific queries for better results
   - Adjust `top_k` based on your use case (5-20 is typically sufficient)
   - Use filters to narrow down search scope when possible
6. **Document Deduplication**: The service uses SHA256 hashing to prevent duplicate documents. Use `force_reindex` only when you need to update an existing document.

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to test endpoints directly from your browser.

### Code Examples Repository

For more examples and SDK implementations, check the project repository or contribute your own client libraries.

## Web UI Usage

### Upload Page (`/`)
1. Navigate to http://localhost:8000
2. Select one or more files (md, txt, json)
3. Optionally add metadata JSON
4. Click "Upload"
5. View upload progress and success/error messages

### Documents List (`/documents/ui`)
- View all ingested documents
- See chunk counts, token counts, and creation dates
- Delete documents individually

### Search Page (`/search/ui`)
1. Enter your search query
2. Select number of results (top_k)
3. Click "Search"
4. View ranked results with scores and chunk previews

## Configuration

Environment variables (see `.env.example`):

- `EMBEDDING_MODEL`: Sentence-transformers model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `CHUNK_SIZE_TOKENS`: Token count per chunk (default: `500`)
- `CHUNK_OVERLAP_TOKENS`: Overlap between chunks (default: `50`)
- `SQLITE_PATH`: Path to SQLite database file
- `QDRANT_HOST`: Qdrant hostname (default: `qdrant` for Docker)
- `QDRANT_PORT`: Qdrant port (default: `6333`)
- `QDRANT_COLLECTION_NAME`: Collection name (default: `documents`)

## Data Model

### SQLite (Metadata)
- `documents` table stores document metadata
- Fields: id, name, content_type, sha256, created_at, updated_at, status, metadata_json, chunk_count, total_tokens

### Qdrant (Vectors)
- Each chunk stored as a vector point
- Payload includes: doc_id, name, content_type, sha256, chunk_index, token_count, chunk_text, metadata_json, created_at

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Test coverage includes:
- Token chunking with overlap verification
- JSON flattening extraction
- Basic API endpoint tests

## VPS Deployment

### Prerequisites
- VPS with Docker and Docker Compose installed
- Domain name (optional, for reverse proxy)

### Steps

1. **Clone repository**:
   ```bash
   git clone <repo-url>
   cd Chunkenizer
   ```

2. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   nano .env  # Edit as needed
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

4. **Set up reverse proxy** (Nginx example):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

5. **Persistent storage**:
   - SQLite database: `./data/chunkenizer.db` (mounted volume)
   - Qdrant storage: Docker volume `qdrant_storage`

### Backup

To backup data:
```bash
# Backup SQLite
cp data/chunkenizer.db backup/

# Backup Qdrant (if needed)
docker exec qdrant qdrant-backup --output-dir /backup
```

## Project Structure

```
Chunkenizer/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── db/
│   │   ├── database.py      # SQLAlchemy setup
│   │   └── models.py        # Database models
│   ├── ingest/
│   │   ├── chunker.py       # Token-based chunking
│   │   └── processor.py     # Document processing
│   ├── vectorstore/
│   │   └── qdrant_client.py # Qdrant integration
│   ├── api/
│   │   └── routes.py        # API endpoints
│   └── ui/
│       ├── routes.py         # UI routes
│       └── templates/        # Jinja2 templates
├── tests/                    # Pytest tests
├── data/                     # SQLite database (created at runtime)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Troubleshooting

### Qdrant Connection Issues
- Ensure Qdrant container is running: `docker ps`
- Check Qdrant logs: `docker logs qdrant`
- Verify QDRANT_HOST environment variable matches your setup

### Embedding Model Download
- First run downloads the model (~90MB)
- Ensure internet connection for initial download
- Model is cached in `~/.cache/huggingface/`

### Database Issues
- Ensure `data/` directory exists and is writable
- Check SQLite file permissions

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

