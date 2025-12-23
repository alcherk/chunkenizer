"""API route handlers."""
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
    # Check SQLite
    sqlite_ok = False
    try:
        db.execute("SELECT 1")
        sqlite_ok = True
    except:
        pass
    
    # Check Qdrant
    qdrant_ok = get_vectorstore().health_check()
    
    return {
        "status": "ok",
        "sqlite": "ok" if sqlite_ok else "error",
        "qdrant": "ok" if qdrant_ok else "error",
        "embedding_model": settings.embedding_model
    }


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    metadata_json: Optional[str] = Form(None),
    force_reindex: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Upload and process a document."""
    try:
        # Read file content
        content = await file.read()
        
        # Check if document already exists
        processor = get_processor()
        vectorstore = get_vectorstore()
        
        sha256 = processor.compute_sha256(content)
        existing_doc = db.query(Document).filter(Document.sha256 == sha256).first()
        
        if existing_doc and not force_reindex:
            return {
                "message": "Document already exists",
                "document_id": existing_doc.id,
                "chunk_count": existing_doc.chunk_count,
                "total_tokens": existing_doc.total_tokens
            }
        
        # Process document
        result = processor.process_document(
            content=content,
            content_type=file.content_type or "text/plain",
            filename=file.filename or "unknown",
            metadata_json=metadata_json
        )
        
        # Delete existing document if reindexing
        if existing_doc:
            vectorstore.delete_by_doc_id(existing_doc.id)
            db.delete(existing_doc)
            db.commit()
        
        # Create document record
        doc = Document(
            name=file.filename or "unknown",
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
        
        # Prepare vectors and payloads for Qdrant
        vectors = result["embeddings"]
        payloads = []
        ids = []
        
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
        vectorstore.add_vectors(vectors, payloads, ids)
        
        return {
            "message": "Document ingested successfully",
            "document_id": doc.id,
            "chunk_count": doc.chunk_count,
            "total_tokens": doc.total_tokens,
            "sha256": sha256
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(
    request: SearchRequest = Body(...),
    db: Session = Depends(get_db)
):
    """Search for similar document chunks."""
    try:
        processor = get_processor()
        vectorstore = get_vectorstore()
        
        # Generate query embedding
        query_embedding = processor.embedding_model.encode(request.query, show_progress_bar=False).tolist()
        
        # Search in Qdrant
        results = vectorstore.search(
            query_vector=query_embedding,
            top_k=request.top_k,
            filters=request.filters
        )
        
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
        
        return {
            "query": request.query,
            "results": formatted_results,
            "total_results": len(formatted_results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    """List all ingested documents."""
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return {
        "documents": [doc.to_dict() for doc in documents],
        "total": len(documents)
    }


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Get document details."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc.to_dict()


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Delete a document and its vectors."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete from Qdrant
    vectorstore = get_vectorstore()
    vectorstore.delete_by_doc_id(doc_id)
    
    # Delete from database
    db.delete(doc)
    db.commit()
    
    return {"message": "Document deleted successfully", "document_id": doc_id}

