# app/retrieval/search_cli.py
"""
Command-line interface for document search functionality.

Note: This CLI currently only supports FAISS-based vector search.
The referenced VectorSearcher, BM25Searcher, and HybridSearcher classes
are not yet implemented in this codebase.
"""

import argparse
import json
import logging

from app.embed.model import Embedder
from app.retrieval import FaissSqliteSearcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main CLI function for document search.

    Currently supports only FAISS-based vector search.
    """
    parser = argparse.ArgumentParser(
        description="Search documents using FAISS vector similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.retrieval.search_cli "machine learning algorithms"
  python -m app.retrieval.search_cli "neural networks" --k 10 --db my_database.db
        """,
    )

    parser.add_argument("query", help="Search query string")
    parser.add_argument("--db", default="rag_local.db", help="Path to SQLite database")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--index-file", help="Path to FAISS index file (default: faiss_index/index.faiss)"
    )
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument(
        "--rerank-k", type=int, default=20, help="Number of candidates for reranking"
    )
    parser.add_argument(
        "--reranker-model", default="BAAI/bge-reranker-base", help="Reranker model name"
    )
    parser.add_argument("--reranker-batch", type=int, default=16, help="Reranker batch size")
    parser.add_argument(
        "--max-passage-chars",
        type=int,
        default=1200,
        help="Max characters per passage for reranking",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize embedder
        logger.info(f"Loading embedding model: {args.model}")
        embedder = Embedder(model_name=args.model)

        # Initialize searcher
        searcher = FaissSqliteSearcher(
            embedder=embedder, db_path=args.db, index_file=args.index_file
        )

        # Perform search
        logger.info(f"Searching for: '{args.query}'")
        hits = searcher.search(args.query, top_k=args.k)

        # Apply reranking if requested
        if args.rerank and hits:
            logger.info("Applying cross-encoder reranking")
            from app.retrieval import Reranker

            reranker = Reranker(
                model_name=args.reranker_model,
                batch_size=args.reranker_batch,
                max_passage_chars=args.max_passage_chars,
            )

            # Get more candidates for reranking
            if len(hits) < args.rerank_k:
                more_hits = searcher.search(args.query, top_k=args.rerank_k)
                hits = more_hits

            hits = reranker.rerank(args.query, hits, top_k=args.k)

        # Output results
        print(json.dumps(hits, ensure_ascii=False, indent=2))

        if not hits:
            logger.warning("No results found for the given query")

    except Exception as e:
        logger.error(f"Search failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
