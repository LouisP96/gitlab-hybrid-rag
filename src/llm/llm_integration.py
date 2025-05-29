from src.retrieval.query_embeddings import RAGRetriever, Reranker, HybridRetriever
import anthropic
import re
from typing import List, Dict, Any, Optional


class QueryRewriter:
    """
    Uses Claude to intelligently rewrite queries based on conversation context.
    """
    
    def __init__(self, client):
        self.client = client
    
    def rewrite_query(self, query: str, conversation_history: List[Dict]) -> str:
        """
        Use Claude to rewrite the query with appropriate context.
        
        Args:
            query: Original user query
            conversation_history: Previous conversation
            
        Returns:
            Rewritten query optimized for retrieval
        """
        if not conversation_history:
            return query
            
        # Build context from recent conversation
        context = ""
        for entry in conversation_history[-2:]:
            context += f"User: {entry['query']}\nAssistant: {entry['answer']}\n\n"
        
        rewrite_prompt = f"""Given this conversation context and the user's new query, rewrite the query to be optimal for code/documentation search.

        CONVERSATION CONTEXT:
        {context}

        USER'S NEW QUERY: {query}

        Instructions:
        1. If this is a follow-up question, expand it with relevant context from the conversation
        2. If this is a completely new topic, return the original query unchanged
        3. Keep technical terms, function names, file names from the context if relevant
        4. Make the query specific and searchable

        Return only the rewritten query, nothing else."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                temperature=0.1,
                messages=[{"role": "user", "content": rewrite_prompt}]
            )
            
            rewritten = response.content[0].text.strip()
            print(f"Query rewritten from '{query}' to '{rewritten}'")
            return rewritten
            
        except Exception as e:
            print(f"Query rewriting failed: {e}, using original query")
            return query


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
        reranker_model_name="BAAI/bge-reranker-v2-m3",
        reranker_batch_size=32,
        # Hybrid parameters
        bm25_index_path="data/embeddings_output/bm25_index.pkl",
        semantic_weight=0.8,
        retrieval_k=25,
        rerank_candidates=20,
    ):
        """
        Initialize the GitLab RAG system with hybrid retrieval.
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
        
        # Initialize query rewriter
        self.query_rewriter = QueryRewriter(self.client)

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
        Ask a question about the codebase with intelligent query rewriting.
        
        Args:
            query: The current question to ask
            conversation_history: Previous conversation for context
            top_k: Number of chunks to retrieve
        """
        # Use Claude to rewrite the query if needed
        retrieval_query = self.query_rewriter.rewrite_query(query, conversation_history or [])
        
        # Retrieve chunks using the (possibly rewritten) query
        chunks = self.retriever.search(retrieval_query, top_k=top_k)

        # Format context for the LLM
        context = "CONTEXT:\n\n"
        for i, chunk in enumerate(chunks):
            filtered_content = self.filter_content(chunk["augmented_content"])
            context += f"[SOURCE {i + 1}]\n{filtered_content}\n\n"

        # Build conversation context for the LLM
        conversation_context = ""
        if conversation_history:
            conversation_context = "CONVERSATION HISTORY:\n"
            for msg in conversation_history[-2:]:
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
        9. Use British English
        10. For follow-up questions, ensure you maintain consistency with previous answers while providing new information

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
