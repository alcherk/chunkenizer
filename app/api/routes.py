"""API route handlers."""
import logging
import traceback
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
import json

from app.db.database import get_db
from app.db.models import Document
from app.ingest.processor import DocumentProcessor
from app.vectorstore.qdrant_client import QdrantStore
from app.config import settings

# Set up logger for this module
logger = logging.getLogger("app.api")

router = APIRouter()

# Initialize processor and vectorstore lazily to avoid issues at import time
_processor = None
_vectorstore = None

def get_processor():
    """Get or create document processor."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor

def get_vectorstore():
    """Get or create vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = QdrantStore()
    return _vectorstore


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    top_k: int = 5
    filters: Optional[dict] = None


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    logger.info("Health check requested")
    
    # Check SQLite
    sqlite_ok = False
    try:
        db.execute("SELECT 1")
        sqlite_ok = True
        logger.debug("SQLite health check: OK")
    except Exception as e:
        logger.error(f"SQLite health check failed: {e}", exc_info=True)
    
    # Check Qdrant
    qdrant_ok = False
    try:
        qdrant_ok = get_vectorstore().health_check()
        logger.debug(f"Qdrant health check: {'OK' if qdrant_ok else 'FAILED'}")
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}", exc_info=True)
    
    status = {
        "status": "ok",
        "sqlite": "ok" if sqlite_ok else "error",
        "qdrant": "ok" if qdrant_ok else "error",
        "embedding_model": settings.embedding_model
    }
    
    logger.info(f"Health check result: {status}")
    return status


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    metadata_json: Optional[str] = Form(None),
    force_reindex: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Upload and process a document."""
    filename = file.filename or "unknown"
    logger.info(f"Document upload requested: {filename} (size: {file.size if hasattr(file, 'size') else 'unknown'}, force_reindex: {force_reindex})")
    
    try:
        # Read file content
        content = await file.read()
        content_size = len(content)
        logger.debug(f"Read {content_size} bytes from {filename}")
        
        # Check if document already exists
        processor = get_processor()
        vectorstore = get_vectorstore()
        
        sha256 = processor.compute_sha256(content)
        logger.debug(f"Document SHA256: {sha256}")
        
        existing_doc = db.query(Document).filter(Document.sha256 == sha256).first()
        
        if existing_doc and not force_reindex:
            logger.info(f"Document already exists: {filename} (doc_id: {existing_doc.id})")
            return {
                "message": "Document already exists",
                "document_id": existing_doc.id,
                "chunk_count": existing_doc.chunk_count,
                "total_tokens": existing_doc.total_tokens
            }
        
        if existing_doc and force_reindex:
            logger.info(f"Force reindex requested for existing document: {filename} (doc_id: {existing_doc.id})")
        
        # Process document
        logger.info(f"Processing document: {filename}")
        result = processor.process_document(
            content=content,
            content_type=file.content_type or "text/plain",
            filename=filename,
            metadata_json=metadata_json
        )
        
        logger.info(f"Document processed: {len(result['chunks'])} chunks, {result['total_tokens']} tokens")
        
        # Delete existing document if reindexing
        if existing_doc:
            logger.info(f"Deleting existing document vectors: {existing_doc.id}")
            vectorstore.delete_by_doc_id(existing_doc.id)
            db.delete(existing_doc)
            db.commit()
            logger.info(f"Deleted existing document: {existing_doc.id}")
        
        # Create document record
        doc = Document(
            name=filename,
            content_type=file.content_type or "text/plain",
            sha256=sha256,
            metadata_json=metadata_json,
            chunk_count=len(result["chunks"]),
            total_tokens=result["total_tokens"],
            status="ingested"
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        logger.info(f"Document record created: {doc.id}")
        
        # Prepare vectors and payloads for Qdrant
        vectors = result["embeddings"]
        payloads = []
        ids = []
        
        logger.debug(f"Preparing {len(vectors)} vectors for Qdrant storage")
        for idx, chunk in enumerate(result["chunks"]):
            payload = {
                "doc_id": doc.id,
                "name": doc.name,
                "content_type": doc.content_type,
                "sha256": doc.sha256,
                "chunk_index": chunk["chunk_index"],
                "token_count": chunk["token_count"],
                "chunk_text": chunk["text"],
                "metadata_json": metadata_json,
                "created_at": datetime.utcnow().isoformat()
            }
            payloads.append(payload)
            ids.append(f"{doc.id}_{chunk['chunk_index']}")
        
        # Store vectors in Qdrant
        logger.info(f"Storing {len(vectors)} vectors in Qdrant")
        vectorstore.add_vectors(vectors, payloads, ids)
        logger.info(f"Successfully stored vectors in Qdrant for document: {doc.id}")
        
        response = {
            "message": "Document ingested successfully",
            "document_id": doc.id,
            "chunk_count": doc.chunk_count,
            "total_tokens": doc.total_tokens,
            "sha256": sha256
        }
        
        logger.info(f"Document upload completed successfully: {filename} (doc_id: {doc.id}, chunks: {doc.chunk_count}, tokens: {doc.total_tokens})")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document {filename}: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(
    request: SearchRequest = Body(...),
    db: Session = Depends(get_db)
):
    """Search for similar document chunks."""
    logger.info(f"Search requested: query='{request.query[:50]}...' (top_k={request.top_k}, filters={request.filters})")
    
    try:
        processor = get_processor()
        vectorstore = get_vectorstore()
        
        # Generate query embedding
        logger.debug("Generating query embedding")
        query_embedding = processor.embedding_model.encode(request.query, show_progress_bar=False).tolist()
        logger.debug(f"Query embedding generated: {len(query_embedding)} dimensions")
        
        # Search in Qdrant
        logger.debug(f"Searching Qdrant with top_k={request.top_k}")
        results = vectorstore.search(
            query_vector=query_embedding,
            top_k=request.top_k,
            filters=request.filters
        )
        logger.info(f"Qdrant search returned {len(results)} results")
        
        # Format results
        formatted_results = []
        for result in results:
            payload = result["payload"]
            formatted_results.append({
                "score": result["score"],
                "document_id": payload.get("doc_id"),
                "document_name": payload.get("name"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_text": payload.get("chunk_text", ""),
                "token_count": payload.get("token_count", 0),
                "metadata": json.loads(payload.get("metadata_json", "{}")) if payload.get("metadata_json") else {}
            })
        
        response = {
            "query": request.query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
        
        logger.info(f"Search completed: {len(formatted_results)} results returned")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    """List all ingested documents."""
    logger.info("Listing all documents")
    
    try:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        logger.info(f"Found {len(documents)} documents")
        
        return {
            "documents": [doc.to_dict() for doc in documents],
            "total": len(documents)
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Get document details."""
    logger.info(f"Getting document details: {doc_id}")
    
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            logger.warning(f"Document not found: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.debug(f"Document found: {doc.name}")
        return doc.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Delete a document and its vectors."""
    logger.info(f"Delete document requested: {doc_id}")
    
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            logger.warning(f"Document not found for deletion: {doc_id}")
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Deleting document: {doc.name} (doc_id: {doc_id})")
        
        # Delete from Qdrant
        vectorstore = get_vectorstore()
        logger.debug(f"Deleting vectors from Qdrant for document: {doc_id}")
        vectorstore.delete_by_doc_id(doc_id)
        logger.debug(f"Vectors deleted from Qdrant for document: {doc_id}")
        
        # Delete from database
        db.delete(doc)
        db.commit()
        logger.info(f"Document deleted successfully: {doc_id}")
        
        return {"message": "Document deleted successfully", "document_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

