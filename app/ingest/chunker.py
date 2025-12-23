"""Token-based text chunking with overlap."""
import tiktoken
from typing import List
from app.config import settings


class TokenChunker:
    """Chunk text using tiktoken tokenizer."""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = settings.chunk_size_tokens
        self.chunk_overlap = settings.chunk_overlap_tokens
    
    def chunk_text(self, text: str) -> List[dict]:
        """
        Split text into chunks with overlap.
        
        Returns:
            List of dicts with keys: 'text', 'token_count', 'chunk_index'
        """
        if not text.strip():
            return []
        
        # Tokenize the text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= self.chunk_size:
            return [{
                "text": text,
                "token_count": len(tokens),
                "chunk_index": 0
            }]
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode tokens back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "chunk_index": chunk_index
            })
            
            # Move start_idx forward, accounting for overlap
            if end_idx >= len(tokens):
                break
            start_idx = end_idx - self.chunk_overlap
            chunk_index += 1
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

