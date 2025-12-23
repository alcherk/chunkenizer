"""Document processing: extraction, chunking, and embedding."""
import hashlib
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from app.ingest.chunker import TokenChunker
from app.config import settings


class DocumentProcessor:
    """Process documents: extract text, chunk, and generate embeddings."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.chunker = TokenChunker()
        self.vector_size = len(self.embedding_model.encode("test"))
    
    def extract_text(self, content: bytes, content_type: str, filename: str) -> str:
        """Extract text from document based on content type."""
        if content_type in ["text/plain", "text/markdown"]:
            return content.decode("utf-8")
        
        elif content_type == "application/json":
            try:
                data = json.loads(content.decode("utf-8"))
                # Flatten JSON to text
                return self._flatten_json(data)
            except json.JSONDecodeError:
                return content.decode("utf-8", errors="ignore")
        
        else:
            # Try to decode as UTF-8 text
            return content.decode("utf-8", errors="ignore")
    
    def _flatten_json(self, obj: Any, prefix: str = "") -> str:
        """Flatten JSON object to text representation."""
        lines = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.extend(self._flatten_json(value, new_key).split("\n"))
                else:
                    lines.append(f"{new_key}: {value}")
        
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                new_key = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                if isinstance(item, (dict, list)):
                    lines.extend(self._flatten_json(item, new_key).split("\n"))
                else:
                    lines.append(f"{new_key}: {item}")
        
        else:
            lines.append(f"{prefix}: {obj}")
        
        return "\n".join(lines)
    
    def compute_sha256(self, content: bytes) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content).hexdigest()
    
    def process_document(
        self,
        content: bytes,
        content_type: str,
        filename: str,
        metadata_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a document: extract text, chunk, and generate embeddings.
        
        Returns:
            Dict with keys: 'text', 'chunks', 'embeddings', 'sha256', 'total_tokens'
        """
        # Extract text
        text = self.extract_text(content, content_type, filename)
        
        # Compute hash
        sha256 = self.compute_sha256(content)
        
        # Chunk text
        chunks = self.chunker.chunk_text(text)
        
        if not chunks:
            raise ValueError("Document produced no chunks")
        
        # Generate embeddings for each chunk
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=False)
        
        # Convert numpy arrays to lists
        embeddings = [emb.tolist() for emb in embeddings]
        
        # Calculate total tokens
        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        
        return {
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings,
            "sha256": sha256,
            "total_tokens": total_tokens,
            "metadata_json": metadata_json
        }

