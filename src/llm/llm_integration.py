from src.retrieval.query_embeddings import RAGRetriever, Reranker, HybridRetriever
import anthropic
import re


class GitLabRAG:
    def __init__(
        self,
        api_key=None,
        filtered_words=None,
        # Vector retriever parameters
        index_path="data/embeddings_output/combined_index.faiss",
        id_mapping_path="data/embeddings_output/combined_id_mapping.json",
        project_mapping_path="data/embeddings_output/chunk_to_project.json",
        chunks_dir="data/enriched_output",
        model_name="Alibaba-NLP/gte-multilingual-base",
        # Reranker parameters
        reranker_model_name="BAAI/bge-reranker-base",
        reranker_batch_size=32,
        # Hybrid parameters
        bm25_index_path="data/embeddings_output/bm25_index.pkl",
        semantic_weight=0.8,
        retrieval_k=25,
        rerank_candidates=20,
    ):
        """
        Initialize the GitLab RAG system with hybrid retrieval.

        Args:
            api_key: API key for the LLM service
            filtered_words: List of words to filter out from chunk content
            Other args: Configuration for vector retriever, reranker, and BM25
        """

        # Create vector retriever (without reranking)
        vector_retriever = RAGRetriever(
            index_path=index_path,
            id_mapping_path=id_mapping_path,
            project_mapping_path=project_mapping_path,
            chunks_dir=chunks_dir,
            model_name=model_name,
            use_reranker=False,
        )

        # Create reranker
        reranker = Reranker(
            model_name=reranker_model_name, batch_size=reranker_batch_size
        )

        # Create hybrid retriever
        self.retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            reranker=reranker,
            bm25_index_path=bm25_index_path,
            semantic_weight=semantic_weight,
            retrieval_k=retrieval_k,
            rerank_candidates=rerank_candidates,
        )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.filtered_words = filtered_words or []

    def filter_content(self, content):
        """Filter out specified words from the content."""
        if not self.filtered_words:
            return content

        pattern = (
            r"\b(" + "|".join(re.escape(word) for word in self.filtered_words) + r")\b"
        )
        filtered_content = re.sub(pattern, "", content, flags=re.IGNORECASE)
        filtered_content = re.sub(r"\s+", " ", filtered_content)

        return filtered_content

    def ask(self, query, conversation_history=None, top_k=10):
        """
        Ask a question about the codebase.
        
        Args:
            query: The current question to ask
            conversation_history: Previous conversation for context (not used for retrieval)
            top_k: Number of chunks to retrieve
        """
        # Use ONLY the current query for retrieval - not the conversation history
        chunks = self.retriever.search(query, top_k=top_k)

        # Format context for the LLM
        context = "CONTEXT:\n\n"
        for i, chunk in enumerate(chunks):
            filtered_content = self.filter_content(chunk["augmented_content"])
            context += f"[SOURCE {i + 1}]\n{filtered_content}\n\n"

        # Build conversation context for the LLM (separate from retrieval)
        conversation_context = ""
        if conversation_history:
            conversation_context = "CONVERSATION HISTORY:\n"
            for msg in conversation_history[-2:]:  # Last 2 exchanges for context
                conversation_context += f"User: {msg['query']}\nAssistant: {msg['answer']}\n\n"
            conversation_context += "---\n\n"

        # Format prompt for Claude
        prompt = f"{conversation_context}{context}\nCURRENT QUESTION: {query}\n\nANSWER:"

        # Create system prompt
        system_prompt = """You are a specialised assistant for a GitLab codebase.
        
        Your primary function is to answer questions about code, documentation, and project structure by referencing the provided context.

        When responding:
        1. Base your answers primarily on the CONTEXT provided, not on prior knowledge
        2. Use the conversation history to understand the context of the current question, but focus on answering the CURRENT QUESTION
        3. If the context doesn't contain relevant information for the current question, say so clearly
        4. Cite specific sources by referring to their source numbers ([SOURCE X])
        5. Use markdown formatting for code snippets and technical explanations
        6. Be concise but thorough in your explanations
        7. Do not hallucinate file paths, function names, or code that isn't in the context
        8. If the answer requires combining information from multiple chunks, explain how they relate

        The context contains chunks specifically retrieved for the current question."""

        # Call Claude
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0.2,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract source information
        sources = []
        for chunk in chunks:
            source_info = {
                "project": chunk["project"],
                "score": chunk["score"],
                "retrieval_method": chunk.get("retrieval_method", "unknown"),
            }

            # Add optional metadata fields if they exist
            metadata_fields = [
                "path",
                "type",
                "language",
                "filename",
                "chunk_type",
                "parent_id",
                "author",
                "issue_id",
            ]

            for field in metadata_fields:
                if field in chunk:
                    source_info[field] = chunk[field]

            if "metadata" in chunk:
                for field in metadata_fields:
                    if field in chunk["metadata"] and field not in source_info:
                        source_info[field] = chunk["metadata"][field]

            sources.append(source_info)

        return {"answer": response.content[0].text, "sources": sources}