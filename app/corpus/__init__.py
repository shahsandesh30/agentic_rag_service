# app/corpus/__init__.py
"""
Corpus processing module for document ingestion and management.

This module provides comprehensive document processing capabilities including:
- File processing and text extraction from various formats (PDF, TXT, MD)
- Text cleaning and normalization
- Intelligent chunking with heading awareness
- Database storage with metadata management
- Command-line interface for batch processing

Main components:
- File processing: Reading and extracting text from documents
- Text cleaning: Normalizing and cleaning extracted text
- Text chunking: Splitting text into semantic chunks
- Database operations: Storing documents and chunks with metadata
- Ingestion pipeline: End-to-end document processing workflow
"""

from .chunk import chunk_text, sliding_window_chunks, split_by_headings
from .clean import clean_whitespace, normalize_text, remove_control_characters
from .files import file_sha256, iter_paths, read_text_any, sniff_mime
from .ingest import ingest_path
from .interfaces import (
    ChunkData,
    DatabaseManager,
    DocumentData,
    DocumentProcessor,
    FileIterator,
    FileMetadata,
    ProcessingStats,
    TextChunker,
    TextCleaner,
)
from .schema import (
    assign_faiss_ids,
    chunk_ids_for_faiss_ids,
    connect,
    fetch_chunk_texts,
    fetch_full_chunks,
    init_db,
    missing_embedding_chunk_ids,
    upsert_chunks,
    upsert_document,
    upsert_embeddings,
)

__all__ = [
    # File processing
    "iter_paths",
    "file_sha256",
    "sniff_mime",
    "read_text_any",
    # Text cleaning
    "normalize_text",
    "clean_whitespace",
    "remove_control_characters",
    # Text chunking
    "chunk_text",
    "split_by_headings",
    "sliding_window_chunks",
    # Database operations
    "connect",
    "init_db",
    "upsert_document",
    "upsert_chunks",
    "upsert_embeddings",
    "fetch_chunk_texts",
    "fetch_full_chunks",
    "missing_embedding_chunk_ids",
    "assign_faiss_ids",
    "chunk_ids_for_faiss_ids",
    # Main ingestion
    "ingest_path",
    # Interfaces and types
    "DocumentProcessor",
    "TextChunker",
    "TextCleaner",
    "FileIterator",
    "DatabaseManager",
    "ChunkData",
    "DocumentData",
    "ProcessingStats",
    "FileMetadata",
]
