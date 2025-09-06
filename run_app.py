import os
import argparse
from src.web.app import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GitLab RAG web application")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Retrieval parameters
    parser.add_argument("--retrieval-top-k", type=int, default=20, help="Total number of candidates to retrieve before reranking")
    
    # Hybrid retriever parameters
    parser.add_argument("--semantic-only", action="store_true", help="Use semantic-only retriever (default: hybrid)")
    parser.add_argument("--semantic-weight", type=float, default=0.8, help="Weight for semantic search (0.0-1.0)")
    parser.add_argument("--per-system-k", type=int, default=15, help="Number of results each system (semantic/BM25) retrieves before fusion")
    parser.add_argument("--bm25-k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter")
    
    # Reranker parameters
    parser.add_argument("--no-reranker", action="store_true", help="Disable reranking (default: enabled)")
    parser.add_argument("--rerank-max-results", type=int, default=10, help="Maximum results to return after reranking")
    parser.add_argument("--reranker-batch-size", type=int, default=32, help="Batch size for reranking")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3", help="Reranker model to use")
    
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Warning: ANTHROPIC_API_KEY not found in environment variables.")

    app = create_app(
        use_hybrid=not args.semantic_only,
        retrieval_top_k=args.retrieval_top_k,
        semantic_weight=args.semantic_weight,
        per_system_k=args.per_system_k,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        use_reranker=not args.no_reranker,
        rerank_max_results=args.rerank_max_results,
        reranker_batch_size=args.reranker_batch_size,
        reranker_model=args.reranker_model
    )

    in_posit = "RS_SERVER_URL" in os.environ and os.environ["RS_SERVER_URL"]

    print(f"\n{'*' * 60}")
    print("GitLab RAG Flask app is running!")
    if in_posit:
        print("Running in Posit Workbench")
        print(f"Access at: Your Posit URL with /p/{args.port}/")
    print(f"Direct access: http://localhost:{args.port}")
    print(f"Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"{'*' * 60}\n")

    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
