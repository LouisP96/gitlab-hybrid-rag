import re
import pickle
import nltk
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


class BM25Index:
    """
    BM25 implementation using rank-bm25 library with code-friendly preprocessing.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        initialise the BM25 index.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        # Code-friendly defaults - no stemming, no stopword removal
        self.stemmer = None
        self.stop_words = set()

        self.bm25 = None
        self.chunk_ids = []
        self.tokenized_corpus = []

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt_tab", quiet=True)

    def preprocess_text(self, text: str) -> List[str]:
        """
        Code-friendly preprocessing - minimal processing to preserve technical terms.

        Args:
            text: Input text to preprocess

        Returns:
            List of processed tokens
        """
        # Basic cleaning
        text = text.lower()

        # Keep programming-related characters
        text = re.sub(r"[^\w\s\-\.\#\+\_]", " ", text)

        # Tokenize
        tokens = word_tokenize(text)

        # Filter tokens - only remove very short ones
        processed_tokens = []
        for token in tokens:
            if len(token) >= 2:  # Keep tokens 2+ characters
                processed_tokens.append(token)

        return processed_tokens

    def build_index(
        self, documents: List[Dict[str, Any]], content_key: str = "content"
    ):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document dictionaries
            content_key: Key in document dict containing the text content
        """
        print(f"Building BM25 index from {len(documents)} documents...")

        self.chunk_ids = []
        self.tokenized_corpus = []

        # Preprocess all documents
        for i, doc in enumerate(documents):
            if i % 1000 == 0:
                print(f"Processing document {i}/{len(documents)}")

            content = doc.get(content_key, "")
            tokens = self.preprocess_text(content)

            self.tokenized_corpus.append(tokens)
            self.chunk_ids.append(doc.get("chunk_id", f"doc_{i}"))

        # Build BM25 index with custom parameters
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        print(f"BM25 index built successfully with {len(self.chunk_ids)} documents")

    def search(self, query: str, per_system_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25.

        Args:
            query: Search query
            per_system_k: Number of results to return

        Returns:
            List of (chunk_id, score) tuples sorted by score descending
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        # Preprocess query
        query_tokens = self.preprocess_text(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Create (chunk_id, score) tuples and sort
        results = [(self.chunk_ids[i], scores[i]) for i in range(len(scores))]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:per_system_k]

    def save_index(self, filepath: str):
        """Save the BM25 index to disk."""
        index_data = {
            "k1": self.k1,
            "b": self.b,
            "chunk_ids": self.chunk_ids,
            "tokenized_corpus": self.tokenized_corpus,
            "bm25_data": {
                "k1": self.bm25.k1,
                "b": self.bm25.b,
                "epsilon": getattr(
                    self.bm25, "epsilon", 0.25
                ),  # Default if not present
                "doc_freqs": self.bm25.doc_freqs,
                "idf": self.bm25.idf,
                "doc_len": self.bm25.doc_len,
                "avgdl": self.bm25.avgdl,
                "corpus_size": self.bm25.corpus_size,
                # nd attribute may not exist in all versions
                "nd": getattr(self.bm25, "nd", None),
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(index_data, f)

        print(f"BM25 index saved to {filepath}")

    def load_index(self, filepath: str):
        """Load the BM25 index from disk."""
        with open(filepath, "rb") as f:
            index_data = pickle.load(f)

        self.k1 = index_data.get("k1", 1.2)
        self.b = index_data.get("b", 0.75)
        self.chunk_ids = index_data["chunk_ids"]
        self.tokenized_corpus = index_data["tokenized_corpus"]

        # Reconstruct BM25 object with saved parameters
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        # Restore BM25 internal state
        bm25_data = index_data["bm25_data"]
        self.bm25.k1 = bm25_data["k1"]
        self.bm25.b = bm25_data["b"]

        # Handle optional attributes that may not exist in all versions
        if "epsilon" in bm25_data:
            self.bm25.epsilon = bm25_data["epsilon"]

        self.bm25.doc_freqs = bm25_data["doc_freqs"]
        self.bm25.idf = bm25_data["idf"]
        self.bm25.doc_len = bm25_data["doc_len"]
        self.bm25.avgdl = bm25_data["avgdl"]
        self.bm25.corpus_size = bm25_data["corpus_size"]

        # nd attribute may not exist in all versions
        if "nd" in bm25_data and bm25_data["nd"] is not None:
            self.bm25.nd = bm25_data["nd"]

        print(f"BM25 index loaded from {filepath}")


def weighted_fusion(
    semantic_rankings: List[Tuple[str, float]],
    bm25_rankings: List[Tuple[str, float]],
    semantic_weight: float = 0.8,
) -> List[Tuple[str, float]]:
    """
    Combine semantic and BM25 rankings using Anthropic's weighted fusion approach.

    Args:
        semantic_rankings: List of (chunk_id, score) from semantic search
        bm25_rankings: List of (chunk_id, score) from BM25 search
        semantic_weight: Weight for semantic search results (default 0.8)

    Returns:
        Combined ranking as list of (chunk_id, combined_score) tuples
    """
    # Get all unique chunk IDs
    semantic_ids = [chunk_id for chunk_id, _ in semantic_rankings]
    bm25_ids = [chunk_id for chunk_id, _ in bm25_rankings]
    all_chunk_ids = list(set(semantic_ids + bm25_ids))

    # Validate semantic_weight
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError(
            f"semantic_weight must be between 0.0 and 1.0, got {semantic_weight}"
        )

    bm25_weight = 1.0 - semantic_weight

    # Calculate combined scores using Anthropic's formula
    chunk_scores = {}

    for chunk_id in all_chunk_ids:
        score = 0.0

        # Add semantic contribution if present
        if chunk_id in semantic_ids:
            rank = semantic_ids.index(chunk_id)
            score += semantic_weight * (1.0 / (rank + 1))

        # Add BM25 contribution if present
        if chunk_id in bm25_ids:
            rank = bm25_ids.index(chunk_id)
            score += bm25_weight * (1.0 / (rank + 1))

        chunk_scores[chunk_id] = score

    # Sort by combined score
    combined_results = [(chunk_id, score) for chunk_id, score in chunk_scores.items()]
    combined_results.sort(key=lambda x: x[1], reverse=True)

    return combined_results