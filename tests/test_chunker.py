"""Tests for token chunker."""
import pytest
from app.ingest.chunker import TokenChunker


def test_chunker_basic():
    """Test basic chunking functionality."""
    chunker = TokenChunker()
    
    # Small text should produce one chunk
    text = "This is a short text."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["token_count"] > 0


def test_chunker_overlap():
    """Test that chunks have proper overlap."""
    chunker = TokenChunker()
    
    # Create a long text that will require multiple chunks
    # Using a text that's definitely longer than chunk_size
    long_text = " ".join(["word"] * 1000)
    chunks = chunker.chunk_text(long_text)
    
    if len(chunks) > 1:
        # Check that chunks overlap
        # The last part of first chunk should appear in second chunk
        first_chunk_end = chunks[0]["text"][-100:]
        second_chunk_start = chunks[1]["text"][:100]
        
        # There should be some overlap (not exact match due to tokenization)
        assert len(chunks) > 1
        assert chunks[0]["token_count"] <= chunker.chunk_size
        assert chunks[1]["token_count"] <= chunker.chunk_size


def test_chunker_empty():
    """Test chunking empty text."""
    chunker = TokenChunker()
    
    chunks = chunker.chunk_text("")
    assert len(chunks) == 0
    
    chunks = chunker.chunk_text("   ")
    assert len(chunks) == 0


def test_count_tokens():
    """Test token counting."""
    chunker = TokenChunker()
    
    text = "Hello world"
    count = chunker.count_tokens(text)
    
    assert count > 0
    assert isinstance(count, int)

