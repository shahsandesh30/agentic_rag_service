from __future__ import annotations

import argparse
import json
from typing import Any

from app.eval.dataset import load_dataset
from app.eval.metrics import (
    answer_similarity,
    citation_alignment,
    faithfulness_proxy,
    mrr,
    ndcg,
    recall_at_k,
    retrieval_labels,
)
from app.llm.hf import HFGenerator
from app.qa.answer import answer_question
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.hybrid import HybridSearcher
from app.retrieval.store import connect, fetch_full_chunks
from app.retrieval.vector import VectorSearcher


def run_eval(
    dataset_path: str,
    db_path: str = "rag_local.db",
    emb_model: str = "BAAI/bge-small-en-v1.5",
    top_k_ctx: int = 8,
    k_vector: int = 40,
    k_bm25: int = 40,
    rrf_k: int = 60,
    rerank: bool = True,
    rerank_k: int = 20,
) -> dict[str, Any]:
    items = load_dataset(dataset_path)
    vs = VectorSearcher(db_path=db_path, model_name=emb_model)
    bs = BM25Searcher(db_path=db_path)
    hs = HybridSearcher(vs, bs, db_path=db_path)
    gen = HFGenerator()

    rows: list[dict[str, Any]] = []

    for it in items:
        qid = it["id"]
        q = it["question"]
        gold = it.get("gold_answer", "")
        gold_rules = it.get("gold_citations", [])

        # 1) retrieval pass
        hits = hs.search(
            q,
            top_k=max(top_k_ctx, 10),
            k_vector=k_vector,
            k_bm25=k_bm25,
            rrf_k=rrf_k,
            rerank=rerank,
            rerank_k=rerank_k,
        )
        retrieved_ids = [h["chunk_id"] for h in hits]
        labels = retrieval_labels(hits, gold_rules)
        r_at5 = recall_at_k(labels, k=5)
        r_at10 = recall_at_k(labels, k=10)
        mrr10 = mrr(labels, k=10)
        ndcg10 = ndcg(labels, k=10)

        conn = connect(db_path)
        full_map = fetch_full_chunks(conn, retrieved_ids[:top_k_ctx])
        conn.close()
        ctx_texts = [full_map.get(cid, "") for cid in retrieved_ids[:top_k_ctx]]

        # 2) grounded answer
        payload = answer_question(
            q,
            hs,
            gen,
            top_k_ctx=top_k_ctx,
            k_vector=k_vector,
            k_bm25=k_bm25,
            rrf_k=rrf_k,
            rerank=rerank,
            rerank_k=rerank_k,
        )
        ans = payload.answer
        cites = [c.dict() for c in payload.citations]

        # 3) metrics
        a_sim = answer_similarity(ans, gold, emb_model=emb_model) if gold else 0.0
        faithful = faithfulness_proxy(ans, ctx_texts, emb_model=emb_model)
        cite_align = citation_alignment(cites, retrieved_ids)

        rows.append(
            {
                "id": qid,
                "retrieval_recall@5": round(r_at5, 3),
                "retrieval_recall@10": round(r_at10, 3),
                "mrr@10": round(mrr10, 3),
                "ndcg@10": round(ndcg10, 3),
                "answer_sim": round(a_sim, 3),
                "faithfulness": round(faithful, 3),
                "citation_align": round(cite_align, 3),
                "confidence": round(payload.confidence, 3),
                "blocked": bool(payload.safety.blocked),
            }
        )

    # summary
    def avg(key):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return round(sum(vals) / max(1, len(vals)), 3)

    summary = {
        "n": len(rows),
        "retrieval_recall@5": avg("retrieval_recall@5"),
        "retrieval_recall@10": avg("retrieval_recall@10"),
        "mrr@10": avg("mrr@10"),
        "ndcg@10": avg("ndcg@10"),
        "answer_sim": avg("answer_sim"),
        "faithfulness": avg("faithfulness"),
        "citation_align": avg("citation_align"),
        "confidence": avg("confidence"),
        "blocked_rate": round(sum(1 for r in rows if r["blocked"]) / max(1, len(rows)), 3),
    }
    return {"rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser(description="Run offline RAG eval.")
    ap.add_argument("--dataset", default="app/eval/samples.yaml")
    ap.add_argument("--db", default="rag_local.db")
    ap.add_argument("--emb-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--top-k-ctx", type=int, default=8)
    ap.add_argument("--k-vector", type=int, default=40)
    ap.add_argument("--k-bm25", type=int, default=40)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--out-json", default="eval_results.json")
    ap.add_argument("--out-csv", default="eval_results.csv")
    args = ap.parse_args()

    out = run_eval(
        dataset_path=args.dataset,
        db_path=args.db,
        emb_model=args.emb_model,
        top_k_ctx=args.top_k_ctx,
        k_vector=args.k_vector,
        k_bm25=args.k_bm25,
        rrf_k=args.rrf_k,
        rerank=args.rerank,
        rerank_k=args.rerank_k,
    )
    print(json.dumps(out["summary"], indent=2))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # rows → CSV
    keys = list(out["rows"][0].keys()) if out["rows"] else []
    if keys:
        import csv

        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(out["rows"])


if __name__ == "__main__":
    main()
