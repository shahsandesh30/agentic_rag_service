# app/retrieval/search_cli.py
import argparse, json
from app.retrieval.vector import VectorSearcher
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.hybrid import HybridSearcher

def main():
    p = argparse.ArgumentParser(description="Search: vector | bm25 | hybrid")
    p.add_argument("query")
    p.add_argument("--db", default="rag_local.db")
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--mode", choices=["vector","bm25","hybrid"], default="hybrid")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--kv", type=int, default=40, help="vector candidates for hybrid")
    p.add_argument("--kb", type=int, default=40, help="bm25 candidates for hybrid")
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-k", type=int, default=20)
    p.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    p.add_argument("--reranker-batch", type=int, default=16)
    p.add_argument("--max-passage-chars", type=int, default=1200)

    args = p.parse_args()

    vs = VectorSearcher(db_path=args.db, model_name=args.model)
    bs = BM25Searcher(db_path=args.db)
    hs = HybridSearcher(vs, bs, db_path=args.db)

    if args.mode == "vector":
        hits = vs.search(args.query, top_k=args.k)
    elif args.mode == "bm25":
        hits = bs.search(args.query, top_k=args.k)
    else:
        hits = hs.search(
            args.query,
            top_k=args.k,
            k_vector=args.kv,
            k_bm25=args.kb,
            rrf_k=args.rrf_k,
            rerank=args.rerank,
            rerank_k=args.rerank_k,
            reranker_model=args.reranker_model,
            reranker_batch=args.reranker_batch,
            max_passage_chars=args.max_passage_chars,
        )
        
    print(json.dumps(hits, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
