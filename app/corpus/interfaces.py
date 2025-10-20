# app/corpus/interfaces.py
"""
Base interfaces and protocols for corpus processing components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterator, Tuple, Optional
from pathlib import Path


class DocumentProcessor(ABC):
    """Abstract base class for document processing operations."""
    
    @abstractmethod
    def process(self, file_path: str) -> Tuple[str, int]:
        """
        Process a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (processed_text, page_count)
        """
        pass


class TextChunker(ABC):
    """Abstract base class for text chunking operations."""
    
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Chunk text into segments.
        
        Args:
            text: Text to chunk
            **kwargs: Additional chunking parameters
            
        Returns:
            List of chunk dictionaries
        """
        pass


class TextCleaner(ABC):
    """Abstract base class for text cleaning operations."""
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        pass


class FileIterator(ABC):
    """Abstract base class for file iteration operations."""
    
    @abstractmethod
    def iterate_files(self, root_path: str) -> Iterator[str]:
        """
        Iterate over supported files in a directory.
        
        Args:
            root_path: Root directory or file path
            
        Yields:
            Paths to supported files
        """
        pass


class DatabaseManager(ABC):
    """Abstract base class for database operations."""
    
    @abstractmethod
    def connect(self, db_path: str) -> Any:
        """Connect to database."""
        pass
    
    @abstractmethod
    def initialize_schema(self, connection: Any) -> None:
        """Initialize database schema."""
        pass
    
    @abstractmethod
    def upsert_document(self, connection: Any, doc_data: Dict[str, Any]) -> None:
        """Upsert document data."""
        pass
    
    @abstractmethod
    def upsert_chunks(self, connection: Any, chunks: List[Dict[str, Any]]) -> None:
        """Upsert chunk data."""
        pass


# Type aliases for better code clarity
ChunkData = Dict[str, Any]
DocumentData = Dict[str, Any]
ProcessingStats = Dict[str, Any]
FileMetadata = Dict[str, Any]
