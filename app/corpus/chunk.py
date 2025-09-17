# app/corpus/chunk.py
import re, hashlib
from typing import List, Dict, Optional

HEADING_RE = re.compile(r"^(#+\s+.+)$", re.MULTILINE)

def _hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]

def split_by_headings(text: str) -> List[Dict]:
    """
    Best-effort heading segmentation for markdown-like content.
    Returns list of {section_title, start, end}.
    """
    sections = []
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [{"section_title": None, "start": 0, "end": len(text)}]
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        sections.append({
            "section_title": m.group(1).strip("# ").strip(),
            "start": start,
            "end": end
        })
    return sections

def sliding_window_chunks(text: str, section: Optional[str],
                          max_chars=1200, overlap=150) -> List[Dict]:
    """Split text into overlapping sliding windows."""
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunk_text = text[i:j]
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "start_char": i,
                "end_char": j,
                "section": section
            })
        if j >= len(text):
            break
        i = max(0, j - overlap)
    return chunks

def chunk_text(text: str, heading_aware=True,
               max_chars=1200, overlap=150) -> List[Dict]:
    """
    Public API used by ingestion.
    Returns list of chunks with {text, start_char, end_char, section}.
    """
    all_chunks = []
    if heading_aware:
        for sec in split_by_headings(text):
            sec_text = text[sec["start"]:sec["end"]]
            all_chunks += sliding_window_chunks(sec_text,
                                                sec["section_title"],
                                                max_chars=max_chars,
                                                overlap=overlap)
    else:
        all_chunks += sliding_window_chunks(text, None,
                                            max_chars=max_chars,
                                            overlap=overlap)
    return all_chunks
