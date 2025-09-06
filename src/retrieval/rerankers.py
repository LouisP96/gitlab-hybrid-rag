import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Any


class Reranker:
    def __init__(
        self, model_name="BAAI/bge-reranker-v2-m3", batch_size=8, max_results=10
    ):
        """
        initialise the reranker.

        Args:
            model_name: The reranking model to use
            batch_size: Batch size for processing
            max_results: Number of top results to return after reranking
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_results = max_results

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        content_key: str = "content",
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query: The search query
            documents: List of document dictionaries to rerank
            content_key: The key in the document dictionaries that contains the text to score

        Returns:
            Reranked list of documents with updated score
        """
        if not documents:
            return []

        # Check if we have fewer documents than max_results
        if len(documents) < self.max_results:
            print(
                f"Warning: Reranker configured for max_results={self.max_results} but only {len(documents)} documents provided"
            )

        # Extract document texts
        texts = [doc[content_key] for doc in documents]

        print("\n=== RERANKER ===")
        print(f"Query length: {len(query)} chars")
        print(f"Query (first 100 chars): {query[:100]}...")
        print(f"Number of chunks to rerank: {len(texts)}")

        for i, text in enumerate(texts):
            print(
                f"Chunk {i + 1}: {len(text)} chars, first 50 chars: {text[:50].replace(chr(10), ' ')}..."
            )

        scores = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]

                print(
                    f"\nProcessing batch starting at index {i}, batch size: {len(batch_texts)}"
                )

                for j, text in enumerate(batch_texts):
                    combined_text = f"{query} {text}"
                    tokens = self.tokenizer.encode(
                        combined_text, add_special_tokens=True
                    )
                    print(f"  Pair {i + j + 1}: Combined length = {len(tokens)} tokens")

                # Tokenize query and documents
                features = self.tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=1024,
                ).to(self.device)

                # Get scores
                batch_scores = self.model(**features).logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores.tolist())

        print("=== END RERANKER ===\n")

        # Create a copy of the documents list to avoid modifying the original
        reranked_docs = documents.copy()

        # Update scores in place
        for i, score in enumerate(scores):
            reranked_docs[i]["score"] = float(score)
            # Add original rank for debugging
            reranked_docs[i]["original_rank"] = i

        # Sort by score in descending order
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k results
        return reranked_docs[: self.max_results]
