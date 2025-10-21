from __future__ import annotations

import math

import numpy as np

from app.embed.model import Embedder


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(a @ b)


def embed_texts(texts: list[str], model: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    emb = Embedder(model_name=model)
    return emb.encode(texts, batch_size=64)


def retrieval_labels(hits: list[dict], gold_rules: list[dict]) -> list[int]:
    """Return binary relevance per hit using weak gold rules."""
    labs = []
    for h in hits:
        ok = False
        t = (h.get("text") or "").lower()
        sec = (h.get("section") or "").lower()
        path = (h.get("path") or "").lower()
        cid = h.get("chunk_id")
        for r in gold_rules:
            if "chunk_id" in r and r["chunk_id"] == cid:
                ok = True
            if "path_contains" in r and r["path_contains"].lower() in path:
                ok = True
            if "section_contains" in r and r["section_contains"].lower() in sec:
                ok = True
            if "text_contains" in r and r["text_contains"].lower() in t:
                ok = True
            if ok:
                break
        labs.append(1 if ok else 0)
    return labs


def recall_at_k(labels: list[int], k: int = 5) -> float:
    return float(any(labels[:k])) if labels else 0.0


def mrr(labels: list[int], k: int = 10) -> float:
    if not labels:
        return 0.0
    for i, l in enumerate(labels[:k], start=1):
        if l == 1:
            return 1.0 / i
    return 0.0


def ndcg(labels: list[int], k: int = 10) -> float:
    # binary gains; DCG with log2 discount
    gains = [l for l in labels[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    # ideal DCG: one relevant at rank 1 (binary)
    idcg = 1.0
    return float(dcg / idcg) if idcg > 0 else 0.0


def answer_similarity(answer: str, gold: str, emb_model="BAAI/bge-small-en-v1.5") -> float:
    vecs = embed_texts([answer, gold], model=emb_model)
    return cosine_sim(vecs[0], vecs[1])


def faithfulness_proxy(
    answer: str, ctx_texts: list[str], emb_model="BAAI/bge-small-en-v1.5"
) -> float:
    """
    Split answer into sentences; each sentence must be supported by some context chunk.
    Score = mean(max_cosine_per_sentence).
    """
    import re

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sents or not ctx_texts:
        return 0.0
    a_vecs = embed_texts(sents, model=emb_model)
    c_vecs = embed_texts(ctx_texts, model=emb_model)
    sims = []
    for i in range(a_vecs.shape[0]):
        sims.append(float(np.max(c_vecs @ a_vecs[i] / (np.linalg.norm(c_vecs, axis=1) + 1e-8))))
    # Normalize to [0,1] like cosine
    sims = [(s + 1) / 2 if -1 <= s <= 1 else max(0.0, min(1.0, s)) for s in sims]
    return float(sum(sims) / len(sims))


def citation_alignment(pred_citations: list[dict], retrieved_ids: list[str]) -> float:
    if not pred_citations:
        return 0.0
    pred_ids = [c.get("chunk_id") for c in pred_citations if c.get("chunk_id")]
    if not pred_ids:
        return 0.0
    hits = sum(1 for cid in pred_ids if cid in retrieved_ids)
    return hits / max(1, len(pred_ids))
