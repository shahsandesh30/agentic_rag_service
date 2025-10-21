# app/corpus/chunk.py
"""
Text chunking utilities for document processing.

This module provides functions for:
- Splitting text into semantic chunks
- Handling heading-aware segmentation
- Creating sliding window chunks
- Generating stable chunk identifiers
"""

import hashlib
import logging
import re

from .interfaces import ChunkData

logger = logging.getLogger(__name__)

# Compiled regex patterns for performance
HEADING_PATTERN = re.compile(r"^(#+\s+.+)$", re.MULTILINE)

# Configuration constants
DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150
DEFAULT_HASH_LENGTH = 24


def _hash_id(text: str) -> str:
    """
    Generate a stable hash ID for text content.

    Args:
        text: Text to hash

    Returns:
        Hexadecimal hash string (truncated to default length)
    """
    try:
        hash_value = hashlib.sha256(text.encode("utf-8")).hexdigest()[:DEFAULT_HASH_LENGTH]
        logger.debug(f"Generated hash ID: {hash_value[:8]}...")
        return hash_value
    except Exception as e:
        logger.error(f"Hash generation failed: {e}")
        # Fallback to simple hash
        return str(hash(text))[:DEFAULT_HASH_LENGTH]


def split_by_headings(text: str) -> list[dict[str, str | None]]:
    """
    Split text into sections based on markdown-style headings.

    Args:
        text: Text to split into sections

    Returns:
        List of section dictionaries with title, start, and end positions

    Raises:
        TypeError: If input is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    if not text.strip():
        logger.warning("Empty text provided for heading segmentation")
        return [{"section_title": None, "start": 0, "end": 0}]

    try:
        sections = []
        matches = list(HEADING_PATTERN.finditer(text))

        if not matches:
            logger.debug("No headings found, treating as single section")
            return [{"section_title": None, "start": 0, "end": len(text)}]

        logger.debug(f"Found {len(matches)} headings in text")

        for i, match in enumerate(matches):
            start = match.start()
            # Determine end position
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            # Clean up heading text
            heading_text = match.group(1).strip("# ").strip()

            sections.append({"section_title": heading_text, "start": start, "end": end})

            logger.debug(f"Section {i + 1}: '{heading_text}' ({start}-{end})")

        return sections

    except Exception as e:
        logger.error(f"Heading segmentation failed: {e}")
        # Fallback to single section
        return [{"section_title": None, "start": 0, "end": len(text)}]


def sliding_window_chunks(
    text: str,
    section: str | None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkData]:
    """
    Split text into overlapping sliding windows.

    Args:
        text: Text to chunk
        section: Section title for the chunks
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunk dictionaries

    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")

    if overlap >= max_chars:
        logger.warning(f"Overlap ({overlap}) >= max_chars ({max_chars}), reducing overlap")
        overlap = max(0, max_chars - 1)

    if not text.strip():
        logger.warning("Empty text provided for chunking")
        return []

    try:
        chunks = []
        i = 0
        chunk_count = 0

        while i < len(text):
            j = min(len(text), i + max_chars)
            chunk_text = text[i:j]

            if chunk_text.strip():
                chunk_data = {
                    "text": chunk_text,
                    "start_char": i,
                    "end_char": j,
                    "section": section,
                }
                chunks.append(chunk_data)
                chunk_count += 1
                logger.debug(f"Created chunk {chunk_count}: {i}-{j} chars")

            if j >= len(text):
                break

            # Move to next position with overlap
            i = max(0, j - overlap)

        logger.info(f"Created {chunk_count} chunks from text ({len(text)} chars)")
        return chunks

    except Exception as e:
        logger.error(f"Sliding window chunking failed: {e}")
        raise


def chunk_text(
    text: str,
    heading_aware: bool = True,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkData]:
    """
    Main API for text chunking with optional heading awareness.

    Args:
        text: Text to chunk
        heading_aware: Whether to respect heading boundaries
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of chunk dictionaries with text, positions, and metadata

    Raises:
        TypeError: If input is not a string
        ValueError: If parameters are invalid
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    if not text.strip():
        logger.warning("Empty text provided for chunking")
        return []

    try:
        all_chunks = []

        if heading_aware:
            logger.debug("Using heading-aware chunking")
            sections = split_by_headings(text)

            for section_idx, section in enumerate(sections):
                section_text = text[section["start"] : section["end"]]
                section_chunks = sliding_window_chunks(
                    section_text, section["section_title"], max_chars=max_chars, overlap=overlap
                )
                all_chunks.extend(section_chunks)
                logger.debug(f"Section '{section['section_title']}': {len(section_chunks)} chunks")
        else:
            logger.debug("Using simple sliding window chunking")
            all_chunks = sliding_window_chunks(text, None, max_chars=max_chars, overlap=overlap)

        logger.info(f"Text chunking completed: {len(all_chunks)} total chunks")
        return all_chunks

    except Exception as e:
        logger.error(f"Text chunking failed: {e}")
        raise
