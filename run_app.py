import os
import argparse
from src.web.app import create_app

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the GitLab RAG web application with hybrid search"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (causes restart)"
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.8,
        help="Weight for semantic search (0.0-1.0), BM25 weight will be 1-semantic_weight",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=25,
        help="Number of results to retrieve from each system (default: 25)",
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=20,
        help="Number of candidates to rerank (default: 20)",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=32,
        help="Batch size for reranking (default: 32)",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model to use (default: BAAI/bge-reranker-v2-m3)",
    )
    args = parser.parse_args()

    # Set environment variables if needed
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Warning: ANTHROPIC_API_KEY not found in environment variables.")

    # Prepare hybrid parameters
    hybrid_params = {
        "semantic_weight": args.semantic_weight,
        "retrieval_k": args.retrieval_k,
        "rerank_candidates": args.rerank_candidates,
        "reranker_batch_size": args.reranker_batch_size,
        "reranker_model": args.reranker_model,
    }

    # Create and run the app with hybrid search parameters
    app = create_app(**hybrid_params)

    # Determine if we're in Posit Workbench
    in_posit = "RS_SERVER_URL" in os.environ and os.environ["RS_SERVER_URL"]

    port = args.port

    print(f"\n{'*' * 80}")
    print("Flask app is running with Hybrid Search!")
    if in_posit:
        print("You are running in Posit Workbench environment")
        print("Try accessing at:")
        print(f"  - Your Posit URL with: /p/{port}/")
    print("Direct URLs (may not work through all proxies):")
    print(f"  - http://localhost:{port}")
    print(f"  - http://127.0.0.1:{port}")
    print("Configuration:")
    print(f"  - Semantic weight: {args.semantic_weight}")
    print(f"  - Retrieval K: {args.retrieval_k}")
    print(f"  - Rerank candidates: {args.rerank_candidates}")
    print(f"  - Reranker batch size: {args.reranker_batch_size}")
    print(f"  - Reranker model: {args.reranker_model}")
    print(f"  - Debug mode: {'ON' if args.debug else 'OFF'}")
    print(f"{'*' * 80}\n")

    # Run with or without debug mode
    app.run(host="0.0.0.0", port=port, debug=args.debug)
