# app/corpus/ingest.py
"""
Document ingestion pipeline for processing and storing documents.

This module provides functions for:
- Processing documents from various sources
- Text cleaning and chunking
- Database storage with metadata
- Progress tracking and error handling
"""

import hashlib
import json
import logging
import os

from .chunk import chunk_text
from .clean import normalize_text
from .files import file_sha256, iter_paths, read_text_any, sniff_mime
from .interfaces import ChunkData, DocumentData, ProcessingStats
from .schema import connect, init_db, upsert_chunks, upsert_document

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_DB_PATH = "rag_local.db"
DEFAULT_SOURCE = "local"
DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150
DEFAULT_DATA_ROOT = "data"


def _doc_id_for(sha256_hex: str) -> str:
    """
    Generate a stable document ID from SHA256 hash.

    Args:
        sha256_hex: SHA256 hash of document content

    Returns:
        Stable document ID (24 characters)
    """
    try:
        doc_id = hashlib.sha256(sha256_hex.encode("utf-8")).hexdigest()[:24]
        logger.debug(f"Generated document ID: {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"Failed to generate document ID: {e}")
        raise


def _chunk_id(doc_id: str, start: int, end: int) -> str:
    """
    Generate a stable chunk ID from document ID and position.

    Args:
        doc_id: Document ID
        start: Start character position
        end: End character position

    Returns:
        Stable chunk ID (24 characters)
    """
    try:
        chunk_id = hashlib.sha256(f"{doc_id}:{start}:{end}".encode()).hexdigest()[:24]
        logger.debug(f"Generated chunk ID: {chunk_id}")
        return chunk_id
    except Exception as e:
        logger.error(f"Failed to generate chunk ID: {e}")
        raise


def ingest_path(
    root: str,
    db_path: str = DEFAULT_DB_PATH,
    source: str = DEFAULT_SOURCE,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    heading_aware: bool = True,
) -> ProcessingStats:
    """
    Ingest documents from a directory or single file into the database.

    Args:
        root: Path to directory or single file to ingest
        db_path: Path to SQLite database file
        source: Source label for documents
        max_chars: Maximum characters per chunk
        overlap: Character overlap between chunks
        heading_aware: Whether to use heading-aware chunking

    Returns:
        Processing statistics dictionary

    Raises:
        FileNotFoundError: If root path doesn't exist
        sqlite3.Error: If database operations fail
        Exception: If document processing fails
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"Path does not exist: {root}")

    logger.info(f"Starting ingestion from: {root}")
    logger.info(
        f"Parameters: max_chars={max_chars}, overlap={overlap}, heading_aware={heading_aware}"
    )

    try:
        # Initialize database
        conn = connect(db_path)
        init_db(conn)

        n_docs = 0
        n_chunks = 0
        processed_files = []
        skipped_files = []

        for file_path in iter_paths(root):
            try:
                logger.info(f"Processing file: {file_path}")

                # Read and process document
                raw_text, n_pages = read_text_any(file_path)
                if not raw_text.strip():
                    logger.warning(f"Skipping empty file: {file_path}")
                    skipped_files.append(file_path)
                    continue

                # Clean and chunk text
                clean_text = normalize_text(raw_text)
                chunks = chunk_text(
                    clean_text, heading_aware=heading_aware, max_chars=max_chars, overlap=overlap
                )

                if not chunks:
                    logger.warning(f"No chunks created for: {file_path}")
                    skipped_files.append(file_path)
                    continue

                # Process document metadata
                doc_metadata = _process_document_metadata(file_path, source)
                doc_id = _get_or_create_document_id(conn, doc_metadata)

                # Store document
                upsert_document(
                    conn,
                    doc_id,
                    doc_metadata["stored_path"],
                    doc_metadata["source_label"],
                    doc_metadata["sha256"],
                    doc_metadata["mime"],
                    n_pages,
                )

                # Process and store chunks
                chunk_rows = _prepare_chunk_rows(doc_id, chunks, doc_metadata)
                upsert_chunks(conn, chunk_rows)

                # Commit transaction
                conn.commit()

                n_docs += 1
                n_chunks += len(chunk_rows)
                processed_files.append(file_path)

                logger.info(f"Successfully processed: {file_path} ({len(chunk_rows)} chunks)")

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                conn.rollback()
                skipped_files.append(file_path)
                continue

        # Final statistics
        stats = {
            "documents": n_docs,
            "chunks": n_chunks,
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "db": db_path,
        }

        logger.info(f"Ingestion completed: {n_docs} documents, {n_chunks} chunks")
        if skipped_files:
            logger.warning(f"Skipped {len(skipped_files)} files: {skipped_files}")

        return stats

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        if "conn" in locals():
            conn.close()


def _process_document_metadata(file_path: str, source: str) -> DocumentData:
    """
    Process document metadata including MIME type, source label, and path.

    Args:
        file_path: Path to the document file
        source: Source label for the document

    Returns:
        Document metadata dictionary
    """
    try:
        # Get MIME type and determine source label
        mime = sniff_mime(file_path)
        if mime == "application/pdf":
            src_label = "pdf"
        elif mime == "text/markdown":
            src_label = "markdown"
        else:
            src_label = "txt"

        # Compute file hash
        sha_hex = file_sha256(file_path)

        # Process path for storage
        abs_path = os.path.abspath(file_path)
        data_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", DEFAULT_DATA_ROOT)
        )

        if abs_path.lower().startswith((data_root + os.sep).lower()):
            rel_path = os.path.relpath(abs_path, data_root)
            stored_path = os.path.join(DEFAULT_DATA_ROOT, rel_path).replace("\\", "/")
        else:
            stored_path = os.path.join(DEFAULT_DATA_ROOT, os.path.basename(abs_path)).replace(
                "\\", "/"
            )

        return {
            "sha256": sha_hex,
            "mime": mime,
            "source_label": src_label,
            "stored_path": stored_path,
            "original_path": file_path,
        }

    except Exception as e:
        logger.error(f"Failed to process document metadata for {file_path}: {e}")
        raise


def _get_or_create_document_id(conn, doc_metadata: DocumentData) -> str:
    """
    Get existing document ID or create new one.

    Args:
        conn: Database connection
        doc_metadata: Document metadata dictionary

    Returns:
        Document ID
    """
    try:
        # Check if document already exists
        row = conn.execute(
            "SELECT id FROM documents WHERE sha256 = ?", (doc_metadata["sha256"],)
        ).fetchone()
        if row:
            logger.debug(f"Found existing document: {row[0]}")
            return row[0]
        else:
            doc_id = _doc_id_for(doc_metadata["sha256"])
            logger.debug(f"Created new document ID: {doc_id}")
            return doc_id

    except Exception as e:
        logger.error(f"Failed to get/create document ID: {e}")
        raise


def _prepare_chunk_rows(
    doc_id: str, chunks: list[ChunkData], doc_metadata: DocumentData
) -> list[tuple]:
    """
    Prepare chunk rows for database insertion.

    Args:
        doc_id: Document ID
        chunks: List of chunk dictionaries
        doc_metadata: Document metadata

    Returns:
        List of chunk tuples for database insertion
    """
    try:
        rows = []
        for i, chunk in enumerate(chunks):
            chunk_id = _chunk_id(doc_id, chunk["start_char"], chunk["end_char"])
            meta = {
                "source": doc_metadata["source_label"],
                "path": doc_metadata["stored_path"],
                "mime": doc_metadata["mime"],
            }

            row = (
                chunk_id,
                doc_id,
                i,
                chunk["text"],
                len(chunk["text"]),
                chunk["start_char"],
                chunk["end_char"],
                chunk["section"],
                json.dumps(meta),
            )
            rows.append(row)

        logger.debug(f"Prepared {len(rows)} chunk rows for document {doc_id}")
        return rows

    except Exception as e:
        logger.error(f"Failed to prepare chunk rows: {e}")
        raise
