"""Qdrant vector store client."""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings


class QdrantStore:
    """Qdrant vector store wrapper."""
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure collection exists with correct vector size."""
        # Get vector size from embedding model (all-MiniLM-L6-v2 has 384 dimensions)
        vector_size = 384
        
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            # Collection might already exist, try to recreate if needed
            try:
                self.client.delete_collection(self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
            except:
                pass
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """Add vectors to the collection."""
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in vectors]
        
        points = [
            PointStruct(id=id_, vector=vector, payload=payload)
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        query_filter = None
        
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                query_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]
    
    def delete_points(self, point_ids: List[str]) -> bool:
        """Delete multiple points by their IDs."""
        if not point_ids:
            return False
        
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
            return True
        except Exception as e:
            raise Exception(f"Failed to delete points: {e}")
    
    def delete_all_points(self) -> int:
        """Delete all points from the collection."""
        try:
            # First, get the total count
            try:
                count_result = self.client.count(collection_name=self.collection_name)
                total_count = count_result.count
            except Exception as e:
                # If count fails, we'll still try to delete
                total_count = None
            
            # If count is 0, nothing to delete
            if total_count == 0:
                return 0
            
            # Try method 1: Scroll and delete by IDs
            all_point_ids = []
            offset = None
            batch_size = 10000
            
            try:
                while True:
                    scroll_result = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=batch_size,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False
                    )
                    
                    points = scroll_result[0]
                    if not points:
                        break
                    
                    batch_ids = [point.id for point in points]
                    all_point_ids.extend(batch_ids)
                    
                    # Get next offset (point ID for next page)
                    if len(scroll_result) > 1:
                        offset = scroll_result[1]
                    else:
                        offset = None
                    
                    # If we got fewer points than the limit, we've reached the end
                    if len(points) < batch_size or offset is None:
                        break
                
                # Delete all points in batches if there are many
                if all_point_ids:
                    # Qdrant can handle large deletions, but let's do it in batches to be safe
                    delete_batch_size = 10000
                    for i in range(0, len(all_point_ids), delete_batch_size):
                        batch = all_point_ids[i:i + delete_batch_size]
                        self.client.delete(
                            collection_name=self.collection_name,
                            points_selector=batch
                        )
                    
                    return len(all_point_ids)
            except Exception as scroll_error:
                # If scroll method fails, try alternative: delete and recreate collection
                import logging
                logging.warning(f"Scroll-based deletion failed: {scroll_error}. Trying collection recreation method.")
                
                # Method 2: Delete and recreate collection (more reliable but loses collection settings)
                try:
                    # Get collection config first
                    collections = self.client.get_collections()
                    collection_exists = any(col.name == self.collection_name for col in collections.collections)
                    
                    if collection_exists:
                        # Get vector size from existing collection or use default
                        vector_size = 384  # Default for all-MiniLM-L6-v2
                        try:
                            # Try to get config via HTTP
                            import requests
                            response = requests.get(
                                f"http://{settings.qdrant_host}:{settings.qdrant_port}/collections/{self.collection_name}",
                                timeout=5
                            )
                            if response.status_code == 200:
                                data = response.json().get('result', {})
                                if 'config' in data:
                                    params = data['config'].get('params', {})
                                    vectors_config = params.get('vectors', {})
                                    if isinstance(vectors_config, dict):
                                        vector_size = vectors_config.get('size', 384)
                        except:
                            pass
                        
                        # Delete and recreate collection
                        self.client.delete_collection(self.collection_name)
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(
                                size=vector_size,
                                distance=Distance.COSINE
                            )
                        )
                        return total_count if total_count else 0
                except Exception as recreate_error:
                    raise Exception(f"Both deletion methods failed. Scroll error: {scroll_error}, Recreate error: {recreate_error}")
            
            # If we found no points but count says there are points, collection might be in inconsistent state
            if total_count and total_count > 0 and len(all_point_ids) == 0:
                import logging
                logging.warning(f"Found {total_count} points in count but scroll returned 0 points. Collection may be in inconsistent state.")
                # Try to recreate collection as last resort
                try:
                    self.client.delete_collection(self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=384,
                            distance=Distance.COSINE
                        )
                    )
                    return total_count
                except:
                    return 0
            
            return 0
        except Exception as e:
            raise Exception(f"Failed to delete all points: {e}")
    
    def delete_by_doc_id(self, doc_id: str):
        """Delete all vectors for a document."""
        from qdrant_client.models import ScrollRequest
        # Scroll to find all points with matching doc_id
        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
            limit=10000  # Adjust if needed
        )
        
        # Extract point IDs
        point_ids = [point.id for point in scroll_result[0]]
        
        # Delete points by IDs
        if point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            collections = self.client.get_collections()
            return True
        except:
            return False

