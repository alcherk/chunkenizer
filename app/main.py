"""Main FastAPI application."""
from fastapi import FastAPI, Request
from app.api.routes import router as api_router
from app.ui.routes import router as ui_router
from app.db.database import init_db
from app.vectorstore.qdrant_client import QdrantStore
from app.config import settings
from app.logging_config import setup_logging

# Set up logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(title="Chunkenizer", description="RAG Ingestion & Retrieval System")

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    import time
    start_time = time.time()
    
    logger.info(f"→ {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"← {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"✗ {request.method} {request.url.path} - Error: {str(e)} - Time: {process_time:.3f}s", exc_info=True)
        raise

# Include routers
app.include_router(api_router, prefix="/api", tags=["api"])
app.include_router(ui_router, tags=["ui"])

# Also include API routes without prefix for direct access
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Initialize database and vector store on startup."""
    logger.info("Starting Chunkenizer application...")
    
    # Initialize SQLite database
    try:
        init_db()
        logger.info(f"✓ Database initialized: {settings.sqlite_path}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize database: {e}", exc_info=True)
        raise
    
    # Initialize Qdrant collection
    try:
        vectorstore = QdrantStore()
        logger.info(f"✓ Qdrant collection ready: {settings.qdrant_collection_name}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Qdrant: {e}", exc_info=True)
        raise
    
    logger.info(f"✓ Embedding model: {settings.embedding_model}")
    logger.info("Application startup complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

