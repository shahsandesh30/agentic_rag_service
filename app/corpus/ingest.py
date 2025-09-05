import os, hashlib, json
from typing import Optional, Dict, List, Tuple
from slugify import slugify

from .files import iter_paths, read_text_any, sniff_mime, file_sha256
from .clean import normalize_text
from .chunk import chunk_text
from .schema import connect, init_db, upsert_document, upsert_chunks

def _doc_id_for(sha256_hex: str) -> str:
    # stable id derived solely from file content hash
    return hashlib.sha256(sha256_hex.encode("utf-8")).hexdigest()[:24]

def _chunk_id(doc_id: str, start: int, end: int) -> str:
    return hashlib.sha256(f"{doc_id}:{start}:{end}".encode("utf-8")).hexdigest()[:24]

def ingest_path(root: str, db_path: str = "rag_local.db", source: str = "local", max_chars=1200, overlap=150, heading_aware=True) -> dict:
    """
    Returns stats dict.
    """
    conn = connect(db_path)
    init_db(conn)
    n_docs = 0
    n_chunks = 0
    for path in iter_paths(root):
        raw_text, n_pages = read_text_any(path)
        if not raw_text.strip():
            continue
        clean = normalize_text(raw_text)
        mime = sniff_mime(path)

        # normalize source label by type
        if mime == "application/pdf":
            src_label = "pdf"
        elif mime == "text/markdown":
            src_label = "markdown"
        else:
            src_label = "txt"

        # store path relative to data/ if present
        abs_path = os.path.abspath(path)
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        stored_path: str
        if abs_path.lower().startswith((data_root + os.sep).lower()):
            rel_path = os.path.relpath(abs_path, data_root)
            stored_path = os.path.join("data", rel_path).replace("\\", "/")
        else:
            stored_path = os.path.join("data", os.path.basename(abs_path)).replace("\\", "/")

        sha_hex = file_sha256(path)
        # reuse existing document if this sha already exists
        row = conn.execute("SELECT id FROM documents WHERE sha256 = ?", (sha_hex,)).fetchone()
        if row:
            doc_id = row[0]
        else:
            doc_id = _doc_id_for(sha_hex)
        upsert_document(conn, doc_id, stored_path, src_label, sha_hex, mime, n_pages)

        chunks = chunk_text(clean, heading_aware=heading_aware, max_chars=max_chars, overlap=overlap)
        rows = []
        for i, ch in enumerate(chunks):
            cid = _chunk_id(doc_id, ch["start_char"], ch["end_char"])
            meta = {"source": src_label, "path": stored_path, "mime": mime}
            rows.append((cid, doc_id, i, ch["text"], len(ch["text"]), ch["start_char"], ch["end_char"], ch["section"], json.dumps(meta)))
        upsert_chunks(conn, rows)
        conn.commit()

        n_docs += 1
        n_chunks += len(rows)

    return {"documents": n_docs, "chunks": n_chunks, "db": db_path}

    
