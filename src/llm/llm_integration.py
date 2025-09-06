import anthropic
import re
from typing import List, Dict


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
            Rewritten query optimised for retrieval
        """
        if not conversation_history:
            return query

        # Build context from recent conversation
        context = ""
        for entry in conversation_history[-2:]:
            context += f"User: {entry['query']}\nAssistant: {entry['answer']}\n\n"

        rewrite_prompt = f"""Given this conversation context and the user's new query, follow the instructions below.

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
                messages=[{"role": "user", "content": rewrite_prompt}],
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
        retriever,
        api_key=None,
        filtered_words=None,
        use_query_rewriter=True,
        reranker=None,
    ):
        """
        initialise the GitLab RAG system.

        Args:
            retriever: Retriever instance (RAGRetriever or HybridRetriever)
            reranker: Optional Reranker instance for reranking results
            use_query_rewriter: Whether to use intelligent query rewriting
        """
        self.retriever = retriever
        self.reranker = reranker

        self.client = anthropic.Anthropic(api_key=api_key)
        self.filtered_words = filtered_words or []

        # initialise query rewriter if requested
        if use_query_rewriter:
            self.query_rewriter = QueryRewriter(self.client)
        else:
            self.query_rewriter = None

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

    def ask(self, query, conversation_history=None, retrieval_top_k=10):
        """
        Ask a question about the codebase with intelligent query rewriting.

        Args:
            query: The current question to ask
            conversation_history: Previous conversation for context
            retrieval_top_k: Number of chunks to retrieve (before optional reranking and further trimming)
        """
        # Use Claude to rewrite the query if needed
        if self.query_rewriter:
            retrieval_query = self.query_rewriter.rewrite_query(
                query, conversation_history or []
            )
        else:
            retrieval_query = query

        # Retrieve chunks using the (possibly rewritten) query
        chunks = self.retriever.search(retrieval_query, retrieval_top_k=retrieval_top_k)

        # Apply reranking if enabled (reranker controls how many to return)
        if self.reranker:
            chunks = self.reranker.rerank(retrieval_query, chunks)

        # Format context for the LLM
        context = "CONTEXT:\n\n"
        for i, chunk in enumerate(chunks):
            filtered_content = self.filter_content(chunk["content"])
            context += f"[SOURCE {i + 1}]\n{filtered_content}\n\n"

        # Build conversation context for the LLM
        conversation_context = ""
        if conversation_history:
            conversation_context = "CONVERSATION HISTORY:\n"
            for msg in conversation_history[-2:]:
                conversation_context += (
                    f"User: {msg['query']}\nAssistant: {msg['answer']}\n\n"
                )
            conversation_context += "---\n\n"

        # Format prompt
        prompt = (
            f"{conversation_context}{context}\nCURRENT QUESTION: {query}\n\nANSWER:"
        )

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
            # Start with basic info, then add all available metadata
            source_info = {
                "project": chunk["project"],
                "score": chunk["score"],
                "retrieval_method": chunk.get("retrieval_method", "unknown"),
            }

            # Add all other fields from the chunk
            for key, value in chunk.items():
                if key not in source_info:  # Don't overwrite basic info
                    source_info[key] = value

            sources.append(source_info)

        return {"answer": response.content[0].text, "sources": sources}
