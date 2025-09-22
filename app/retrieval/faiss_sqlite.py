# app/retrieval/faiss_sqlite.py
from __future__ import annotations
import os
import faiss
import numpy as np
from typing import List, Dict

from app.embed.model import Embedder
from app.corpus.schema import (
    connect,
    chunk_ids_for_faiss_ids,
    fetch_chunk_texts
)

INDEX_DIR = "faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")


class FaissSqliteSearcher:
    def __init__(self, embedder: Embedder, db_path: str = "rag_local.db"):
        self.embedder = embedder
        self.db_path = db_path
        self.index = None
        self._load_faiss_index()

    def _load_faiss_index(self):
        if not os.path.exists(INDEX_FILE):
            print(f"[WARNING] FAISS index not found at {INDEX_FILE}. You must run ingest_and_index.py first.")
            return
        print(f"[INFO] Loading FAISS index from {INDEX_FILE}")
        index = faiss.read_index(INDEX_FILE)

        # Ensure ID support for index.add_with_ids() and search with mapped IDs
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)

        self.index = index

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Run ingest_and_index.py to build it.")

        # Step 1: Embed the query
        q = self.embedder.encode([query]).astype("float32")

        # Step 2: Normalize the query vector (for cosine similarity)
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

        # Step 3: Search in FAISS
        D, I = self.index.search(q, top_k)
        faiss_ids = [int(x) for x in I[0] if x != -1]
        scores = [float(s) for s in D[0][:len(faiss_ids)]]

        if not faiss_ids:
            print("[INFO] No relevant FAISS results found.")
            return []

        # Step 4: Map FAISS IDs to chunk IDs and fetch texts
        conn = connect(self.db_path)
        id_map = chunk_ids_for_faiss_ids(conn, faiss_ids)
        chunk_ids = [id_map.get(fid, "") for fid in faiss_ids]
        meta = fetch_chunk_texts(conn, [cid for cid in chunk_ids if cid])
        conn.close()

        # Step 5: Format final results
        hits: List[Dict] = []
        for fid, score in zip(faiss_ids, scores):
            cid = id_map.get(fid)
            if not cid:
                continue
            m = meta.get(cid, {})
            hits.append({
                "chunk_id": cid,                          # Internal reference
                "score": score,                           # Similarity score
                "text": (m.get("text") or "")[:800],      # Trimmed preview
                "path": m.get("path", ""),                # File path
                "section": m.get("section", ""),          # Optional section title
                "source": m.get("source", ""),            # Optional source type
            })

        return hits
