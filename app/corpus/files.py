import os, mimetypes, hashlib
from typing import Iterator, Tuple

SUPPORTED_EXT = {".pdf", ".txt", ".md"}

def iter_paths(root: str) -> Iterator[str]:
    if os.path.isfile(root):
        yield root 
        return
    
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p  = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXT:
                yield p


def file_sha256(path: str) -> str:
    h  = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()

def sniff_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf": return "application/pdf"
    if ext == ".md": return "text/markdown"
    return mimetypes.guess_type(path)[0] or "text/plain"

def read_text_any(path: str) -> Tuple[str, int]:
    """
    Returns (text, n_pages). For non-PDF, n_pages=1.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for pg in pdf.pages:
                pages.append(pg.extract_text() or "")
        return "\n".join(pages), len(pages)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), 1

