"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
from app.ui.routes import router as ui_router
from app.db.database import init_db
from app.vectorstore.qdrant_client import QdrantStore
from app.config import settings

# Initialize FastAPI app
app = FastAPI(title="Chunkenizer", description="RAG Ingestion & Retrieval System")

# Include routers
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(ui_router, tags=["ui"])

# Also include API routes without prefix for direct access
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Initialize database and vector store on startup."""
    # Initialize SQLite database
    init_db()
    
    # Initialize Qdrant collection
    vectorstore = QdrantStore()
    print(f"✓ Database initialized: {settings.sqlite_path}")
    print(f"✓ Qdrant collection ready: {settings.qdrant_collection_name}")
    print(f"✓ Embedding model: {settings.embedding_model}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

