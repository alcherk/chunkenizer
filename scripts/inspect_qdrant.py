#!/usr/bin/env python3
"""Script to inspect the Qdrant vector database."""
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vectorstore.qdrant_client import QdrantStore
from app.config import settings
from app.db.database import SessionLocal
from app.db.models import Document


def show_collection_info(store):
    """Show collection information."""
    try:
        collections = store.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        print("="*80)
        print("QDRANT COLLECTIONS")
        print("="*80)
        for name in collection_names:
            try:
                # Try to get collection info - may fail due to version compatibility
                info = store.client.get_collection(name)
                print(f"\nCollection: {name}")
                print(f"  Points count: {info.points_count:,}")
                if hasattr(info, 'vectors_count'):
                    print(f"  Vectors count: {info.vectors_count:,}")
                if hasattr(info, 'config') and hasattr(info.config, 'params'):
                    if hasattr(info.config.params, 'vectors'):
                        print(f"  Vector size: {info.config.params.vectors.size}")
                        print(f"  Distance: {info.config.params.vectors.distance}")
            except Exception as e:
                # Fallback: use raw API call
                print(f"\nCollection: {name}")
                try:
                    # Use raw HTTP API to get collection info
                    import requests
                    response = requests.get(f"http://{store.client._host}:{store.client._port}/collections/{name}")
                    if response.status_code == 200:
                        data = response.json()['result']
                        print(f"  Points count: {data.get('points_count', 'unknown'):,}")
                        if 'config' in data:
                            vectors_config = data['config'].get('params', {}).get('vectors', {})
                            if isinstance(vectors_config, dict) and 'size' in vectors_config:
                                print(f"  Vector size: {vectors_config['size']}")
                            elif hasattr(vectors_config, 'size'):
                                print(f"  Vector size: {vectors_config.size}")
                except Exception as e2:
                    print(f"  (Could not retrieve details: {e2})")
        
        return collection_names
    except Exception as e:
        print(f"Error getting collection info: {e}")
        import traceback
        traceback.print_exc()
        return []


def show_sample_points(store, collection_name, limit=5):
    """Show sample points from the collection."""
    try:
        from qdrant_client.models import ScrollRequest
        
        print("\n" + "="*80)
        print(f"SAMPLE POINTS (first {limit})")
        print("="*80)
        
        # Scroll to get points
        scroll_result = store.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False  # Don't print full vectors (too large)
        )
        
        points = scroll_result[0]
        
        if not points:
            print("No points found in collection")
            return
        
        for i, point in enumerate(points, 1):
            print(f"\n--- Point {i} ---")
            print(f"ID: {point.id}")
            print(f"Payload:")
            payload = point.payload
            for key, value in payload.items():
                if key == 'chunk_text':
                    # Truncate long text
                    text_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"  {key}: {text_preview}")
                elif key == 'metadata_json' and value:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
            
            if hasattr(point, 'vector') and point.vector:
                print(f"Vector: [{len(point.vector)} dimensions] (hidden for brevity)")
        
    except Exception as e:
        print(f"Error getting sample points: {e}")
        import traceback
        traceback.print_exc()


def show_document_vectors(store, doc_id):
    """Show all vectors for a specific document."""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        print("\n" + "="*80)
        print(f"VECTORS FOR DOCUMENT: {doc_id}")
        print("="*80)
        
        scroll_result = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_result[0]
        
        if not points:
            print(f"No vectors found for document {doc_id}")
            return
        
        print(f"\nFound {len(points)} chunks/vectors for this document:\n")
        
        for point in points:
            payload = point.payload
            print(f"Chunk {payload.get('chunk_index', '?')}: "
                  f"{payload.get('token_count', 0)} tokens")
            print(f"  Text preview: {payload.get('chunk_text', '')[:80]}...")
            print()
        
    except Exception as e:
        print(f"Error getting document vectors: {e}")
        import traceback
        traceback.print_exc()


def show_statistics(store, db_session):
    """Show statistics about stored vectors."""
    try:
        # Try to get collection info, with fallback
        total_points = 0
        vector_size = 384  # Default for all-MiniLM-L6-v2
        distance = "Cosine"
        
        try:
            info = store.client.get_collection(store.collection_name)
            total_points = info.points_count
            if hasattr(info, 'config') and hasattr(info.config, 'params'):
                if hasattr(info.config.params, 'vectors'):
                    vector_size = info.config.params.vectors.size
                    distance = str(info.config.params.vectors.distance)
        except Exception:
            # Fallback: use raw API
            try:
                import requests
                response = requests.get(f"http://{store.client._host}:{store.client._port}/collections/{store.collection_name}")
                if response.status_code == 200:
                    data = response.json()['result']
                    total_points = data.get('points_count', 0)
                    vectors_config = data.get('config', {}).get('params', {}).get('vectors', {})
                    if isinstance(vectors_config, dict):
                        vector_size = vectors_config.get('size', 384)
                        distance = vectors_config.get('distance', 'Cosine')
            except Exception as e:
                print(f"Warning: Could not get collection info: {e}")
        
        # Get document count from SQLite
        documents = db_session.query(Document).all()
        doc_count = len(documents)
        
        print("\n" + "="*80)
        print("VECTOR DATABASE STATISTICS")
        print("="*80)
        print(f"Total vectors/chunks stored: {total_points:,}")
        print(f"Total documents: {doc_count}")
        if doc_count > 0 and total_points > 0:
            avg_chunks = total_points / doc_count
            print(f"Average chunks per document: {avg_chunks:.2f}")
        print(f"Vector dimensions: {vector_size}")
        print(f"Distance metric: {distance}")
        print("="*80)
        
    except Exception as e:
        print(f"Error getting statistics: {e}")
        import traceback
        traceback.print_exc()


def search_example(store, query_text="machine learning", top_k=3):
    """Perform an example search."""
    try:
        from app.ingest.processor import DocumentProcessor
        
        print("\n" + "="*80)
        print(f"EXAMPLE SEARCH: '{query_text}'")
        print("="*80)
        
        processor = DocumentProcessor()
        query_embedding = processor.embedding_model.encode(query_text, show_progress_bar=False).tolist()
        
        results = store.search(
            query_vector=query_embedding,
            top_k=top_k,
            filters=None
        )
        
        if not results:
            print("No results found")
            return
        
        print(f"\nFound {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            payload = result["payload"]
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Document: {payload.get('name', 'unknown')}")
            print(f"   Chunk: {payload.get('chunk_index', '?')}")
            print(f"   Text: {payload.get('chunk_text', '')[:100]}...")
            print()
        
    except Exception as e:
        print(f"Error performing search: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    print("="*80)
    print("QDRANT VECTOR DATABASE INSPECTOR")
    print("="*80)
    print(f"Qdrant Host: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"Collection: {settings.qdrant_collection_name}")
    print()
    
    try:
        store = QdrantStore()
        db_session = SessionLocal()
        
        # Show collection info
        collections = show_collection_info(store)
        
        if settings.qdrant_collection_name not in collections:
            print(f"\nWarning: Collection '{settings.qdrant_collection_name}' not found!")
            return
        
        # Show statistics
        show_statistics(store, db_session)
        
        # Show sample points
        show_sample_points(store, settings.qdrant_collection_name, limit=3)
        
        # Interactive menu
        while True:
            print("\n" + "="*80)
            print("OPTIONS")
            print("="*80)
            print("1. Show sample vectors")
            print("2. Show vectors for a specific document")
            print("3. Perform example search")
            print("4. Show statistics")
            print("5. Exit")
            print()
            
            choice = input("Enter choice (1-5): ").strip()
            
            if choice == "1":
                limit = input("How many samples? (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                show_sample_points(store, settings.qdrant_collection_name, limit=limit)
            
            elif choice == "2":
                doc_id = input("Enter document ID: ").strip()
                if doc_id:
                    show_document_vectors(store, doc_id)
                else:
                    print("Invalid document ID")
            
            elif choice == "3":
                query = input("Enter search query (default 'machine learning'): ").strip()
                query = query if query else "machine learning"
                top_k = input("Number of results (default 3): ").strip()
                top_k = int(top_k) if top_k.isdigit() else 3
                search_example(store, query, top_k)
            
            elif choice == "4":
                show_statistics(store, db_session)
            
            elif choice == "5":
                print("Exiting...")
                break
            
            else:
                print("Invalid choice")
        
        db_session.close()
    
    except Exception as e:
        print(f"\nError connecting to Qdrant: {e}")
        print("\nMake sure Qdrant is running:")
        print("  - Local: docker run -d -p 6333:6333 qdrant/qdrant:latest")
        print("  - Docker Compose: docker-compose up -d")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

