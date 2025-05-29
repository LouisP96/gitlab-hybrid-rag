from flask import Flask, request, jsonify, render_template
import os
import traceback
import logging
import sys
from flask_cors import CORS


def create_app(
    rag_instance=None,
    semantic_weight=0.8,
    retrieval_k=25,
    rerank_candidates=20,
    reranker_batch_size=8,
    reranker_model="BAAI/bge-reranker-v2-m3",
    **other_kwargs,
):
    """Create and configure the Flask application

    Args:
        rag_instance: Optional pre-initialized RAG instance
        semantic_weight: Weight for semantic search (0.0 to 1.0), BM25 weight will be 1-semantic_weight
        retrieval_k: Number of results to retrieve from each system
        rerank_candidates: Number of candidates to rerank
        reranker_batch_size: Batch size for reranking
        reranker_model: Name of reranker model to use
    """
    app = Flask(__name__)
    CORS(app)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Add Posit Workbench compatibility with improved configuration
    if "RS_SERVER_URL" in os.environ and os.environ["RS_SERVER_URL"]:
        from werkzeug.middleware.proxy_fix import ProxyFix

        app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1, x_host=1, x_port=1, x_proto=1)

    # Use provided RAG instance or create a new one
    if rag_instance is None:
        try:
            from src.llm.llm_integration import GitLabRAG

            api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables!")

            logger.info("Initializing GitLabRAG...")
            rag = GitLabRAG(
                api_key=api_key,
                index_path="data/embeddings_output/combined_index.faiss",
                id_mapping_path="data/embeddings_output/combined_id_mapping.json",
                project_mapping_path="data/embeddings_output/chunk_to_project.json",
                chunks_dir="data/enriched_output",
                filtered_words=None,
                semantic_weight=semantic_weight,
                retrieval_k=retrieval_k,
                reranker_model_name=reranker_model,
                rerank_candidates=rerank_candidates,
                reranker_batch_size=reranker_batch_size,
            )
            logger.info("GitLabRAG initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            logger.error(traceback.format_exc())
            rag = None
    else:
        rag = rag_instance

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/ask", methods=["POST"])
    def ask():
        try:
            print("Ask endpoint called!")
            # Check if RAG system is available
            if rag is None:
                return jsonify(
                    {
                        "error": "RAG system not initialized properly",
                        "answer": "Sorry, the RAG system is not available. Please check server logs.",
                        "sources": [],
                    }
                ), 500

            # Parse request
            if not request.is_json:
                logger.error(f"Invalid request content type: {request.content_type}")
                return jsonify(
                    {
                        "error": "Invalid request, expected JSON",
                        "answer": "Sorry, there was a problem with your request. Please try again.",
                        "sources": [],
                    }
                ), 400

            data = request.json
            query = data.get("query", "")

            if not query.strip():
                return jsonify(
                    {
                        "error": "Empty query",
                        "answer": "Please provide a question to ask.",
                        "sources": [],
                    }
                ), 400

            conversation_history = data.get("history", [])

            # Log the incoming request
            logger.info(f"Received query: {query}")
            logger.info(f"History length: {len(conversation_history)}")

            # Get response from RAG system - pass history separately
            logger.info("Sending query to RAG system...")
            response = rag.ask(query, conversation_history=conversation_history, top_k=10)
            logger.info("Received response from RAG system")

            # Add CORS headers
            resp = jsonify(
                {
                    "answer": response["answer"],
                    "sources": response.get("sources", []),
                    "hybrid_search_enabled": True,  
                    "query_rewriter_enabled": True, 
                }
            )
            return resp

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify(
                {
                    "error": str(e),
                    "answer": "Sorry, an error occurred while processing your request. Please check server logs.",
                    "sources": [],
                }
            ), 500

    @app.route("/debug", methods=["GET"])
    def debug():
        """Simple endpoint to test if API is functioning"""
        try:
            print("Debug endpoint called!")
            # Check if RAG system is available
            rag_status = "initialized" if rag is not None else "not initialized"

            # Get hybrid search status if available
            hybrid_status = "unknown"
            hybrid_config = {}
            if rag is not None and hasattr(rag, "retriever"):
                try:
                    hybrid_status = "enabled"
                    hybrid_config = {
                        "semantic_weight": semantic_weight,
                        "retrieval_k": retrieval_k,
                        "rerank_candidates": rerank_candidates,
                        "reranker_model": reranker_model,
                        "query_rewriter": True, 
                    }
                except Exception as e:
                    hybrid_status = f"error: {str(e)}"

            return jsonify(
                {
                    "status": "ok",
                    "message": "API is functioning",
                    "rag_status": rag_status,
                    "hybrid_search": hybrid_status,
                    "hybrid_config": hybrid_config,
                    "environment": {
                        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
                        "in_posit": bool(
                            "RS_SERVER_URL" in os.environ
                            and os.environ["RS_SERVER_URL"]
                        ),
                        "flask_env": app.config.get("ENV", "production"),
                        "python_version": sys.version,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error in debug endpoint: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"status": "error", "message": str(e)}), 500

    return app
