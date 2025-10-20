# app/corpus/clean.py
"""
Text cleaning and normalization utilities.

This module provides functions for:
- Normalizing text formatting
- Cleaning whitespace and line endings
- Removing unwanted characters
"""
import re
import logging
from typing import Optional

from .interfaces import TextCleaner

logger = logging.getLogger(__name__)

# Compiled regex patterns for performance
_WHITESPACE_PATTERN = re.compile(r"[ \t]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")
_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_EXTRA_SPACES_PATTERN = re.compile(r" +")


def normalize_text(text: str) -> str:
    """
    Normalize and clean text content.
    
    Performs the following operations:
    1. Normalize line endings (CRLF, CR -> LF)
    2. Remove BOM and control characters
    3. Collapse excessive whitespace
    4. Trim lines and remove empty lines
    5. Collapse multiple newlines
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Cleaned and normalized text
        
    Raises:
        TypeError: If input is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")
    
    if not text:
        logger.debug("Empty text provided for normalization")
        return ""
    
    try:
        # Step 1: Normalize line endings and remove BOM
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
        logger.debug("Normalized line endings and removed BOM")
        
        # Step 2: Remove control characters (except newlines and tabs)
        normalized = _CONTROL_CHARS_PATTERN.sub("", normalized)
        logger.debug("Removed control characters")
        
        # Step 3: Collapse excessive inline whitespace
        normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
        logger.debug("Collapsed excessive whitespace")
        
        # Step 4: Trim lines and remove completely empty lines
        lines = [line.strip() for line in normalized.split("\n")]
        lines = [line for line in lines if line]  # Remove empty lines
        normalized = "\n".join(lines)
        logger.debug("Trimmed lines and removed empty lines")
        
        # Step 5: Collapse multiple newlines to double newlines
        normalized = _MULTI_NEWLINE_PATTERN.sub("\n\n", normalized)
        logger.debug("Collapsed multiple newlines")
        
        # Final trim
        result = normalized.strip()
        
        logger.info(f"Text normalization completed: {len(text)} -> {len(result)} characters")
        return result
        
    except Exception as e:
        logger.error(f"Text normalization failed: {e}")
        # Return original text if normalization fails
        return text.strip()


def clean_whitespace(text: str, preserve_structure: bool = True) -> str:
    """
    Clean whitespace while optionally preserving document structure.
    
    Args:
        text: Text to clean
        preserve_structure: If True, preserve paragraph breaks
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    try:
        if preserve_structure:
            # Preserve paragraph structure
            lines = text.split("\n")
            cleaned_lines = []
            for line in lines:
                # Clean each line but preserve empty lines
                cleaned_line = _EXTRA_SPACES_PATTERN.sub(" ", line.strip())
                cleaned_lines.append(cleaned_line)
            return "\n".join(cleaned_lines)
        else:
            # Aggressive whitespace cleaning
            return _EXTRA_SPACES_PATTERN.sub(" ", text.strip())
            
    except Exception as e:
        logger.error(f"Whitespace cleaning failed: {e}")
        return text


def remove_control_characters(text: str) -> str:
    """
    Remove control characters from text.
    
    Args:
        text: Text to clean
        
    Returns:
        Text with control characters removed
    """
    if not text:
        return ""
    
    try:
        cleaned = _CONTROL_CHARS_PATTERN.sub("", text)
        logger.debug(f"Removed control characters: {len(text)} -> {len(cleaned)} characters")
        return cleaned
    except Exception as e:
        logger.error(f"Control character removal failed: {e}")
        return text
