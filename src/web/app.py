from flask import Flask, request, jsonify, render_template
import os
import traceback
import logging
from flask_cors import CORS


def create_app(
    rag_instance=None,
    use_hybrid=True,
    retrieval_top_k=15,
    semantic_weight=0.8,
    per_system_k=25,
    bm25_k1=1.2,
    bm25_b=0.75,
    use_reranker=True,
    rerank_max_results=10,
    reranker_batch_size=32,
    reranker_model="BAAI/bge-reranker-v2-m3"
):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Add Posit Workbench compatibility
    if "RS_SERVER_URL" in os.environ and os.environ["RS_SERVER_URL"]:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1, x_host=1, x_port=1, x_proto=1)

    # initialise RAG system
    if rag_instance is None:
        try:
            from src.llm.llm_integration import GitLabRAG
            from src.retrieval.retrievers import HybridRetriever, RAGRetriever
            from src.retrieval.rerankers import Reranker

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables!")

            # Choose retriever type
            if use_hybrid:
                logger.info("Initialising HybridRetriever...")
                retriever = HybridRetriever(
                    semantic_weight=semantic_weight,
                    per_system_k=per_system_k,
                    bm25_k1=bm25_k1,
                    bm25_b=bm25_b
                )
            else:
                logger.info("Initialising RAGRetriever (semantic-only)...")
                retriever = RAGRetriever()

            # Initialise reranker if requested
            reranker = None
            if use_reranker:
                logger.info("Initialising reranker...")
                reranker = Reranker(
                    model_name=reranker_model,
                    batch_size=reranker_batch_size,
                    max_results=rerank_max_results
                )
            else:
                logger.info("Reranker disabled")
            
            logger.info("Initialising GitLabRAG...")
            rag = GitLabRAG(
                retriever=retriever,
                api_key=api_key,
                reranker=reranker,
                use_query_rewriter=True
            )
            logger.info("GitLabRAG initialised successfully!")
        except Exception as e:
            logger.error(f"Error initialising RAG system: {str(e)}")
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
            if rag is None:
                return jsonify({
                    "error": "RAG system not initialised",
                    "answer": "RAG system not available",
                    "sources": []
                }), 500

            if not request.is_json:
                return jsonify({
                    "error": "Invalid request format",
                    "answer": "Please send JSON data",
                    "sources": []
                }), 400

            data = request.json
            query = data.get("query", "").strip()

            if not query:
                return jsonify({
                    "error": "Empty query",
                    "answer": "Please provide a question",
                    "sources": []
                }), 400

            conversation_history = data.get("history", [])

            response = rag.ask(query, conversation_history=conversation_history, retrieval_top_k=retrieval_top_k)

            return jsonify({
                "answer": response["answer"],
                "sources": response.get("sources", [])
            })

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return jsonify({
                "error": str(e),
                "answer": "An error occurred while processing your request",
                "sources": []
            }), 500

    return app
