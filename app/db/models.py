"""SQLAlchemy models for document metadata."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text
from app.db.database import Base


class Document(Base):
    """Document metadata model."""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    sha256 = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    status = Column(String, default="ingested", nullable=False)
    metadata_json = Column(Text, nullable=True)
    chunk_count = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "metadata_json": self.metadata_json,
            "chunk_count": self.chunk_count,
            "total_tokens": self.total_tokens,
        }

