# app/retrieval/bm25.py
from __future__ import annotations
from typing import List
import re, numpy as np
from rank_bm25 import BM25Okapi
from app.retrieval.store import connect, load_all_chunks, fetch_chunk_texts

# Optional NLTK tokenizer; fallback to regex if punkt is absent
try:
    from nltk.tokenize import wordpunct_tokenize
    _USE_NLTK = True
except Exception:
    _USE_NLTK = False

def _tok(text: str) -> list[str]:
    if _USE_NLTK:
        try:
            return [t.lower() for t in wordpunct_tokenize(text)]
        except LookupError:
            # fallback if NLTK still complains
            pass
    return re.findall(r"\b\w+\b", text.lower())


class BM25Searcher:
    """
    In-memory BM25 over all chunk texts.
    Suitable for tens of thousands of chunks. Rebuilds fast.
    """
    def __init__(self, db_path: str = "rag_local.db"):
        self.db_path = db_path
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self.reload()

    def reload(self):
        conn = connect(self.db_path)
        self._ids, self._docs = load_all_chunks(conn)
        conn.close()
        self._tokens = [_tok(d) for d in self._docs]
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._bm25:
            return []
        qtok = _tok(query)
        scores = self._bm25.get_scores(qtok)  
        k = min(top_k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        # hydrate metadata
        hit_ids = [self._ids[i] for i in idx.tolist()]
        conn = connect(self.db_path)
        meta_map = fetch_chunk_texts(conn, hit_ids)
        conn.close()

        out = []
        for i in idx.tolist():
            cid = self._ids[i]
            m = meta_map.get(cid, {})
            out.append({
                "chunk_id": cid,
                "score": float(scores[i]),  # BM25 score (not cosine)
                "section": m.get("section"),
                "source": m.get("source"),
                "path": m.get("path"),
                "text": (m.get("text") or "")[:400]
            })
        return out
