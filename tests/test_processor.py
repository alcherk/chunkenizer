"""Tests for document processor."""
import pytest
import json
from app.ingest.processor import DocumentProcessor


def test_json_flatten():
    """Test JSON flattening functionality."""
    processor = DocumentProcessor()
    
    # Test simple JSON
    json_obj = {
        "name": "test",
        "value": 123,
        "nested": {
            "key": "value"
        }
    }
    
    flattened = processor._flatten_json(json_obj)
    
    assert "name: test" in flattened
    assert "value: 123" in flattened
    assert "nested.key: value" in flattened or "nested" in flattened


def test_extract_text_plain():
    """Test plain text extraction."""
    processor = DocumentProcessor()
    
    content = b"Hello world"
    text = processor.extract_text(content, "text/plain", "test.txt")
    
    assert text == "Hello world"


def test_extract_text_json():
    """Test JSON text extraction."""
    processor = DocumentProcessor()
    
    json_content = json.dumps({"key": "value"}).encode("utf-8")
    text = processor.extract_text(json_content, "application/json", "test.json")
    
    assert "key" in text
    assert "value" in text


def test_compute_sha256():
    """Test SHA256 computation."""
    processor = DocumentProcessor()
    
    content = b"test content"
    hash1 = processor.compute_sha256(content)
    hash2 = processor.compute_sha256(content)
    
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 produces 64 hex characters

