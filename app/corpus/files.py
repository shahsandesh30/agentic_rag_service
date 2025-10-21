# app/corpus/files.py
"""
File processing utilities for document ingestion.

This module provides functions for:
- Iterating over supported file types
- Computing file checksums
- Detecting MIME types
- Reading text content from various file formats
"""

import hashlib
import logging
import mimetypes
import os
from collections.abc import Iterator

logger = logging.getLogger(__name__)

# Configuration constants
SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md"}
DEFAULT_CHUNK_SIZE = 1 << 20  # 1MB chunks for file reading
DEFAULT_ENCODING = "utf-8"
DEFAULT_ERROR_HANDLING = "ignore"


def iter_paths(root: str) -> Iterator[str]:
    """
    Iterate over supported files in a directory or return single file.

    Args:
        root: Root directory path or single file path

    Yields:
        Paths to supported files

    Raises:
        FileNotFoundError: If root path doesn't exist
        PermissionError: If access to path is denied
    """
    if not os.path.exists(root):
        raise FileNotFoundError(f"Path does not exist: {root}")

    if not os.access(root, os.R_OK):
        raise PermissionError(f"Permission denied: {root}")

    if os.path.isfile(root):
        ext = os.path.splitext(root)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            logger.debug(f"Found supported file: {root}")
            yield root
        else:
            logger.warning(f"Unsupported file type: {root} (extension: {ext})")
        return

    try:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                ext = os.path.splitext(name)[1].lower()

                if ext in SUPPORTED_EXTENSIONS:
                    logger.debug(f"Found supported file: {file_path}")
                    yield file_path
                else:
                    logger.debug(f"Skipping unsupported file: {file_path} (extension: {ext})")

    except PermissionError as e:
        logger.error(f"Permission denied accessing directory {root}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error iterating over files in {root}: {e}")
        raise


def file_sha256(path: str) -> str:
    """
    Compute SHA256 hash of a file.

    Args:
        path: Path to the file

    Returns:
        SHA256 hash as hexadecimal string

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
        OSError: If file reading fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read file: {path}")

    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(DEFAULT_CHUNK_SIZE), b""):
                hasher.update(chunk)

        hash_value = hasher.hexdigest()
        logger.debug(f"Computed SHA256 for {path}: {hash_value[:16]}...")
        return hash_value

    except Exception as e:
        logger.error(f"Failed to compute SHA256 for {path}: {e}")
        raise


def sniff_mime(path: str) -> str:
    """
    Detect MIME type of a file based on extension.

    Args:
        path: Path to the file

    Returns:
        MIME type string
    """
    ext = os.path.splitext(path)[1].lower()

    # Handle known extensions explicitly
    if ext == ".pdf":
        return "application/pdf"
    elif ext == ".md":
        return "text/markdown"
    elif ext == ".txt":
        return "text/plain"

    # Fall back to mimetypes module
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type:
        return mime_type

    # Default fallback
    logger.warning(f"Unknown MIME type for {path}, defaulting to text/plain")
    return "text/plain"


def read_text_any(path: str) -> tuple[str, int]:
    """
    Read text content from various file formats.

    Args:
        path: Path to the file

    Returns:
        Tuple of (text_content, page_count)

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
        ImportError: If required libraries are missing
        Exception: If text extraction fails
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read file: {path}")

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".pdf":
            return _read_pdf_text(path)
        else:
            return _read_text_file(path)

    except Exception as e:
        logger.error(f"Failed to read text from {path}: {e}")
        raise


def _read_pdf_text(path: str) -> tuple[str, int]:
    """Read text from PDF file."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF processing. Install with: pip install pdfplumber"
        )

    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text() or ""
                    pages.append(text)
                    logger.debug(f"Extracted text from PDF page {page_num + 1}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from PDF page {page_num + 1}: {e}")
                    pages.append("")

        full_text = "\n".join(pages)
        logger.info(f"Extracted {len(pages)} pages from PDF: {path}")
        return full_text, len(pages)

    except Exception as e:
        logger.error(f"PDF processing failed for {path}: {e}")
        raise


def _read_text_file(path: str) -> tuple[str, int]:
    """Read text from plain text file."""
    try:
        with open(path, encoding=DEFAULT_ENCODING, errors=DEFAULT_ERROR_HANDLING) as f:
            content = f.read()
            logger.info(f"Read text file: {path} ({len(content)} characters)")
            return content, 1

    except UnicodeDecodeError as e:
        logger.warning(f"Unicode decode error for {path}, trying with different encoding: {e}")
        try:
            with open(path, encoding="latin-1", errors=DEFAULT_ERROR_HANDLING) as f:
                content = f.read()
                logger.info(f"Read text file with latin-1 encoding: {path}")
                return content, 1
        except Exception as e2:
            logger.error(f"Failed to read {path} with any encoding: {e2}")
            raise
    except Exception as e:
        logger.error(f"Failed to read text file {path}: {e}")
        raise
