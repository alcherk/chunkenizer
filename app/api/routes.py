"""API route handlers."""
import logging
import traceback
import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, text
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
        db.execute(text("SELECT 1"))
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
            # Generate UUID for each point (Qdrant requires valid UUID or integer)
            # We can't use {doc_id}_{chunk_index} format as it's not a valid UUID
            point_id = str(uuid.uuid4())
            ids.append(point_id)
        
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
        
        logger.info(f"Deleting document: {doc.name} (doc_id: {doc_id}, chunks: {doc.chunk_count})")
        
        # Delete from Qdrant first
        qdrant_deleted = False
        try:
            vectorstore = get_vectorstore()
            logger.debug(f"Deleting vectors from Qdrant for document: {doc_id}")
            vectorstore.delete_by_doc_id(doc_id)
            qdrant_deleted = True
            logger.info(f"Successfully deleted {doc.chunk_count} chunks from Qdrant for document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting vectors from Qdrant for document {doc_id}: {e}", exc_info=True)
            # Continue with database deletion even if Qdrant deletion fails
            # This prevents orphaned database records
        
        # Delete from database
        try:
            db.delete(doc)
            db.commit()
            logger.info(f"Document deleted from database: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document from database {doc_id}: {e}", exc_info=True)
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to delete document from database: {str(e)}")
        
        if not qdrant_deleted:
            logger.warning(f"Document {doc_id} deleted from database but Qdrant cleanup failed. Manual cleanup may be required.")
            return {
                "message": "Document deleted from database, but some chunks may remain in Qdrant",
                "document_id": doc_id,
                "warning": "Qdrant cleanup failed"
            }
        
        logger.info(f"Document deleted successfully: {doc_id}")
        return {"message": "Document and all chunks deleted successfully", "document_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


class DeleteDocumentsRequest(BaseModel):
    """Request model for deleting multiple documents."""
    document_ids: List[str]


@router.delete("/documents")
async def delete_documents(
    request: DeleteDocumentsRequest = Body(...),
    db: Session = Depends(get_db)
):
    """Delete multiple documents and their vectors."""
    logger.info(f"Bulk delete requested: {len(request.document_ids)} documents")
    
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")
    
    deleted_docs = []
    failed_docs = []
    
    for doc_id in request.document_ids:
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                logger.warning(f"Document not found for deletion: {doc_id}")
                failed_docs.append({"doc_id": doc_id, "error": "Document not found"})
                continue
            
            logger.info(f"Deleting document: {doc.name} (doc_id: {doc_id}, chunks: {doc.chunk_count})")
            
            # Delete from Qdrant first
            qdrant_deleted = False
            try:
                vectorstore = get_vectorstore()
                logger.debug(f"Deleting vectors from Qdrant for document: {doc_id}")
                vectorstore.delete_by_doc_id(doc_id)
                qdrant_deleted = True
                logger.info(f"Successfully deleted {doc.chunk_count} chunks from Qdrant for document: {doc_id}")
            except Exception as e:
                logger.error(f"Error deleting vectors from Qdrant for document {doc_id}: {e}", exc_info=True)
                # Continue with database deletion even if Qdrant deletion fails
            
            # Delete from database
            try:
                db.delete(doc)
                db.commit()
                logger.info(f"Document deleted from database: {doc_id}")
                
                deleted_docs.append({
                    "doc_id": doc_id,
                    "name": doc.name,
                    "chunks_deleted": doc.chunk_count,
                    "qdrant_cleaned": qdrant_deleted
                })
            except Exception as e:
                logger.error(f"Error deleting document from database {doc_id}: {e}", exc_info=True)
                db.rollback()
                failed_docs.append({"doc_id": doc_id, "name": doc.name, "error": str(e)})
        
        except Exception as e:
            logger.error(f"Unexpected error deleting document {doc_id}: {e}", exc_info=True)
            failed_docs.append({"doc_id": doc_id, "error": str(e)})
    
    response = {
        "message": f"Deleted {len(deleted_docs)} document(s)",
        "deleted_count": len(deleted_docs),
        "failed_count": len(failed_docs),
        "deleted_documents": deleted_docs
    }
    
    if failed_docs:
        response["failed_documents"] = failed_docs
        response["warning"] = f"Some documents failed to delete: {len(failed_docs)}"
    
    return response


@router.get("/qdrant/stats")
async def get_qdrant_stats(db: Session = Depends(get_db)):
    """Get Qdrant collection statistics."""
    logger.info("Qdrant stats requested")
    
    try:
        vectorstore = get_vectorstore()
        
        # Get point count
        count_result = vectorstore.client.count(vectorstore.collection_name)
        total_points = count_result.count
        
        # Get document count from SQLite
        doc_count = db.query(Document).count()
        
        # Try to get collection info via HTTP API
        vector_size = 384
        distance = "COSINE"
        try:
            import requests
            response = requests.get(
                f"http://{settings.qdrant_host}:{settings.qdrant_port}/collections/{vectorstore.collection_name}",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json().get('result', {})
                if 'config' in data:
                    params = data['config'].get('params', {})
                    vectors_config = params.get('vectors', {})
                    if isinstance(vectors_config, dict):
                        vector_size = vectors_config.get('size', 384)
                        distance = vectors_config.get('distance', 'COSINE')
        except:
            pass
        
        return {
            "collection_name": vectorstore.collection_name,
            "total_points": total_points,
            "total_documents": doc_count,
            "vector_size": vector_size,
            "distance_metric": distance,
            "average_chunks_per_doc": round(total_points / doc_count, 2) if doc_count > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error getting Qdrant stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qdrant/points")
async def get_qdrant_points(
    limit: int = 20,
    offset: int = 0,
    doc_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get points from Qdrant collection."""
    logger.info(f"Qdrant points requested: limit={limit}, offset={offset}, doc_id={doc_id}")
    
    try:
        vectorstore = get_vectorstore()
        
        # Build filter if doc_id provided
        scroll_filter = None
        if doc_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            scroll_filter = Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            )
        
        # Scroll to get points
        # Note: Qdrant scroll uses offset as point ID for pagination, not integer offset
        # For initial load, use None. For subsequent pages, use the last point ID from previous result
        scroll_offset = None
        
        # If offset is provided and it's not a doc_id filter, try to use it as point ID
        # For now, we'll use simple scrolling without offset-based pagination
        # Instead, we'll use limit-based pagination which is simpler
        
        scroll_result = vectorstore.client.scroll(
            collection_name=vectorstore.collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            offset=scroll_offset,  # Use None for first page, point ID for next pages
            with_payload=True,
            with_vectors=False  # Don't return full vectors (too large)
        )
        
        points = scroll_result[0]
        # Qdrant returns (points, next_page_offset) tuple
        # next_page_offset is either None or a point ID to use for next page
        next_offset = scroll_result[1] if len(scroll_result) > 1 else None
        
        # Check for data mismatch
        db_doc_count = db.query(Document).count()
        total_db_chunks = db.query(func.sum(Document.chunk_count)).scalar() or 0
        
        if len(points) == 0 and total_db_chunks > 0:
            logger.warning(
                f"Qdrant has 0 points but database shows {db_doc_count} documents "
                f"with {total_db_chunks} total chunks. Vectors may not have been stored."
            )
        
        formatted_points = []
        for point in points:
            payload = point.payload
            formatted_points.append({
                "id": str(point.id),
                "doc_id": payload.get("doc_id"),
                "document_name": payload.get("name"),
                "chunk_index": payload.get("chunk_index"),
                "token_count": payload.get("token_count", 0),
                "chunk_text_preview": payload.get("chunk_text", "")[:200] + "..." if len(payload.get("chunk_text", "")) > 200 else payload.get("chunk_text", ""),
                "created_at": payload.get("created_at")
            })
        
        response = {
            "points": formatted_points,
            "total": len(formatted_points),
            "next_offset": next_offset,
            "has_more": next_offset is not None
        }
        
        # Add warning if there's a mismatch
        if len(points) == 0 and total_db_chunks > 0:
            response["warning"] = (
                f"No vectors found in Qdrant, but database has {db_doc_count} documents "
                f"with {total_db_chunks} chunks. Please re-upload documents to populate vectors."
            )
        
        return response
    except Exception as e:
        logger.error(f"Error getting Qdrant points: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qdrant/points/{point_id}")
async def get_qdrant_point(point_id: str, db: Session = Depends(get_db)):
    """Get a specific point from Qdrant."""
    logger.info(f"Qdrant point requested: {point_id}")
    
    try:
        vectorstore = get_vectorstore()
        
        # Retrieve point by ID
        result = vectorstore.client.retrieve(
            collection_name=vectorstore.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True
        )
        
        if not result:
            raise HTTPException(status_code=404, detail="Point not found")
        
        point = result[0]
        payload = point.payload
        
        # Handle vector - it might be a list or numpy array
        vector_preview = []
        if point.vector:
            if hasattr(point.vector, 'tolist'):
                # It's a numpy array
                vector_preview = point.vector[:10].tolist()
            elif isinstance(point.vector, list):
                # It's already a list
                vector_preview = point.vector[:10]
            else:
                # Try to convert to list
                vector_preview = list(point.vector[:10])
        
        return {
            "id": str(point.id),
            "vector_dimensions": len(point.vector) if point.vector else 0,
            "vector_preview": vector_preview,  # First 10 dimensions
            "payload": {
                "doc_id": payload.get("doc_id"),
                "name": payload.get("name"),
                "content_type": payload.get("content_type"),
                "sha256": payload.get("sha256"),
                "chunk_index": payload.get("chunk_index"),
                "token_count": payload.get("token_count", 0),
                "chunk_text": payload.get("chunk_text", ""),
                "metadata_json": payload.get("metadata_json"),
                "created_at": payload.get("created_at")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting Qdrant point {point_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class DeletePointsRequest(BaseModel):
    """Request model for deleting multiple points."""
    point_ids: List[str]


@router.delete("/qdrant/points")
async def delete_qdrant_points(
    request: DeletePointsRequest = Body(...),
    db: Session = Depends(get_db)
):
    """Delete multiple points from Qdrant and update document metadata."""
    logger.info(f"Bulk delete requested: {len(request.point_ids)} points")
    
    if not request.point_ids:
        raise HTTPException(status_code=400, detail="No point IDs provided")
    
    try:
        vectorstore = get_vectorstore()
        
        # First, retrieve points to get their payloads (for doc_id mapping)
        # We need this before deletion to update document metadata
        points_to_delete = []
        try:
            result = vectorstore.client.retrieve(
                collection_name=vectorstore.collection_name,
                ids=request.point_ids,
                with_payload=True,
                with_vectors=False
            )
            points_to_delete = result
        except Exception as e:
            logger.warning(f"Could not retrieve some points before deletion: {e}")
        
        # Group points by doc_id for metadata updates
        doc_points_map = {}  # {doc_id: [point_ids]}
        for point in points_to_delete:
            doc_id = point.payload.get("doc_id")
            if doc_id:
                if doc_id not in doc_points_map:
                    doc_points_map[doc_id] = []
                doc_points_map[doc_id].append({
                    "point_id": str(point.id),
                    "token_count": point.payload.get("token_count", 0)
                })
        
        # Delete points from Qdrant
        try:
            vectorstore.delete_points(request.point_ids)
            logger.info(f"Successfully deleted {len(request.point_ids)} points from Qdrant")
        except Exception as e:
            logger.error(f"Error deleting points from Qdrant: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to delete points: {str(e)}")
        
        # Update document metadata in SQLite
        updated_docs = []
        for doc_id, points_info in doc_points_map.items():
            try:
                doc = db.query(Document).filter(Document.id == doc_id).first()
                if not doc:
                    logger.warning(f"Document {doc_id} not found in database, skipping metadata update")
                    continue
                
                # Get remaining chunks for this document from Qdrant
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                scroll_result = vectorstore.client.scroll(
                    collection_name=vectorstore.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                        ]
                    ),
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
                
                remaining_points = scroll_result[0]
                new_chunk_count = len(remaining_points)
                new_total_tokens = sum(
                    point.payload.get("token_count", 0) 
                    for point in remaining_points
                )
                
                # Update document record
                doc.chunk_count = new_chunk_count
                doc.total_tokens = new_total_tokens
                doc.updated_at = datetime.utcnow()
                
                # If all chunks deleted, keep status as "ingested" (document still exists)
                db.commit()
                
                updated_docs.append({
                    "doc_id": doc_id,
                    "name": doc.name,
                    "new_chunk_count": new_chunk_count,
                    "new_total_tokens": new_total_tokens
                })
                
                logger.info(
                    f"Updated document {doc_id}: {new_chunk_count} chunks, "
                    f"{new_total_tokens} tokens (deleted {len(points_info)} chunks)"
                )
                
            except Exception as e:
                logger.error(f"Error updating document {doc_id} metadata: {e}", exc_info=True)
                # Continue with other documents even if one fails
                db.rollback()
        
        return {
            "message": "Points deleted successfully",
            "deleted_count": len(request.point_ids),
            "updated_documents": updated_docs,
            "total_documents_updated": len(updated_docs)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting Qdrant points: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/qdrant/points/all")
async def delete_all_qdrant_points(db: Session = Depends(get_db)):
    """Delete all points from Qdrant collection and update document metadata."""
    logger.warning("DELETE ALL POINTS requested - this will remove all vectors from Qdrant")
    
    try:
        vectorstore = get_vectorstore()
        
        # Get count before deletion for logging
        try:
            count_result = vectorstore.client.count(collection_name=vectorstore.collection_name)
            total_points = count_result.count
            logger.info(f"Qdrant collection has {total_points} points before deletion")
        except Exception as e:
            logger.warning(f"Could not get point count before deletion: {e}")
            total_points = 0
        
        # Delete all points
        logger.info("Starting deletion of all points from Qdrant")
        deleted_count = vectorstore.delete_all_points()
        logger.warning(f"Deleted {deleted_count} points from Qdrant collection (expected {total_points})")
        
        # Verify deletion
        try:
            count_result_after = vectorstore.client.count(collection_name=vectorstore.collection_name)
            remaining_points = count_result_after.count
            if remaining_points > 0:
                logger.warning(f"Warning: {remaining_points} points still remain in Qdrant after deletion attempt")
        except Exception as e:
            logger.warning(f"Could not verify deletion: {e}")
        
        # Update all documents to have 0 chunks and 0 tokens
        try:
            documents = db.query(Document).all()
            updated_count = 0
            for doc in documents:
                doc.chunk_count = 0
                doc.total_tokens = 0
                doc.updated_at = datetime.utcnow()
                updated_count += 1
            
            db.commit()
            logger.info(f"Updated {updated_count} documents to have 0 chunks and 0 tokens")
        except Exception as e:
            logger.error(f"Error updating document metadata after deleting all points: {e}", exc_info=True)
            db.rollback()
            # Don't fail the request if metadata update fails, but log it
        
        return {
            "message": "All points deleted from Qdrant successfully and document metadata updated",
            "deleted_count": deleted_count,
            "total_points_before": total_points,
            "documents_updated": updated_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting all Qdrant points: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

