import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import json
import faiss
import re
import pickle
import nltk
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


class RAGRetriever:
    def __init__(
        self,
        index_path="data/embeddings_output/combined_index.faiss",
        id_mapping_path="data/embeddings_output/combined_id_mapping.json",
        project_mapping_path="data/embeddings_output/chunk_to_project.json",
        chunks_dir="data/enriched_output",
        model_name="Alibaba-NLP/gte-multilingual-base",
        use_reranker=False,
        reranker_model_name="BAAI/bge-reranker-base",
    ):
        # Load index
        self.index = faiss.read_index(index_path)

        # Load mappings
        with open(id_mapping_path, "r") as f:
            self.id_mapping = json.load(f)

        with open(project_mapping_path, "r") as f:
            self.chunk_to_project = json.load(f)

        # Set paths
        self.chunks_dir = Path(chunks_dir)

        # Load model with optimizations
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        transformer_model = self.model._first_module().auto_model
        transformer_model.config.unpad_inputs = True
        transformer_model.half()

        # Initialize reranker if requested
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            self.reranker = Reranker(model_name=reranker_model_name)

    def search(
        self,
        query,
        top_k=5,
        filter_project=None,
        metadata_fields=None,
        rerank_top_k=None,
    ):
        """
        Search for relevant chunks.

        Args:
            query: The query text to search for
            top_k: Number of results to return
            filter_project: Optional project name to filter results
            metadata_fields: List of metadata fields to include (default: includes all)
            rerank_top_k: Number of results to rerank (default: 3*top_k if reranking is enabled)

        Returns:
            List of chunk results with metadata
        """
        # Default metadata fields to retrieve
        if metadata_fields is None:
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

        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Determine how many results to retrieve from the vector store
        if self.use_reranker:
            # If reranking, retrieve more results than needed for final output
            search_k = rerank_top_k if rerank_top_k is not None else top_k * 3
        else:
            # If filtering by project, retrieve more to ensure we have enough after filtering
            search_k = top_k * 3 if filter_project else top_k

        # Search the index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), search_k
        )

        # Process results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.id_mapping):
                chunk_id = self.id_mapping[idx]
                project = self.chunk_to_project.get(chunk_id, "unknown")

                # Apply project filter if specified
                if filter_project and project != filter_project:
                    continue

                # Direct file access using chunk_id and project
                file_path = self.chunks_dir / project / f"{chunk_id}.json"

                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)

                    # Start with base result fields
                    result = {
                        "chunk_id": chunk_id,
                        "score": float(dist),
                        "project": project,
                        "augmented_content": chunk_data.get("augmented_content", ""),
                    }

                    # Add metadata fields if they exist
                    metadata = chunk_data.get("metadata", {})
                    for field in metadata_fields:
                        if field in metadata:
                            result[field] = metadata[field]

                    # Add result to list
                    results.append(result)

        # Apply reranking if enabled
        if self.use_reranker and self.reranker and results:
            results = self.reranker.rerank(query, results)

        # Return top_k results
        return results[:top_k]


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-base", batch_size=8):
        """
        Initialize the reranker.

        Args:
            model_name: The reranking model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.batch_size = batch_size

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        content_key: str = "augmented_content",
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

        # Extract document texts
        texts = [doc[content_key] for doc in documents]

        scores = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]

                # Tokenize query and documents
                features = self.tokenizer(
                    [query] * len(batch_texts),
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)

                # Get scores
                batch_scores = self.model(**features).logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores.tolist())

        # Create a copy of the documents list to avoid modifying the original
        reranked_docs = documents.copy()

        # Update scores in place
        for i, score in enumerate(scores):
            reranked_docs[i]["score"] = float(score)
            # Add original rank for debugging
            reranked_docs[i]["original_rank"] = i

        # Sort by score in descending order
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)

        return reranked_docs


class BM25Index:
    """
    BM25 implementation using rank-bm25 library with code-friendly preprocessing.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize the BM25 index.

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
        self, documents: List[Dict[str, Any]], content_key: str = "augmented_content"
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

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

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

        return results[:top_k]

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


class HybridRetriever:
    """
    Hybrid retriever that combines semantic search, BM25, and reranking.
    Always uses: Vector Search + BM25 + Anthropic RRF + Reranking
    """

    def __init__(
        self,
        vector_retriever,  # Existing RAGRetriever (without reranking)
        reranker,
        bm25_index_path: str = "data/embeddings_output/bm25_index.pkl",
        semantic_weight: float = 0.8,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        retrieval_k: int = 50,  # How many to retrieve from each system
        rerank_candidates: int = 30,  # How many candidates to rerank
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: RAGRetriever instance (should not have reranking enabled)
            reranker: Reranker instance
            bm25_index_path: Path to BM25 index
            semantic_weight: Weight for semantic search in fusion (default 0.8)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            retrieval_k: Number of results to retrieve from each system (default 50)
            rerank_candidates: Number of candidates to rerank (default 30)
        """
        self.vector_retriever = vector_retriever
        self.reranker = reranker
        self.semantic_weight = semantic_weight
        self.retrieval_k = retrieval_k
        self.rerank_candidates = rerank_candidates

        # Initialize BM25 index
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)

        # Load or build BM25 index
        if Path(bm25_index_path).exists():
            print(f"Loading BM25 index from {bm25_index_path}")
            self.bm25_index.load_index(bm25_index_path)
        else:
            raise FileNotFoundError(f"BM25 index not found at {bm25_index_path}. ")

    def _build_bm25_index(self, save_path: str):
        """Build BM25 index from chunk files."""
        chunks_dir = Path(self.vector_retriever.chunks_dir)
        documents = []

        print("Loading documents for BM25 indexing...")

        for project_dir in chunks_dir.iterdir():
            if project_dir.is_dir():
                for json_file in project_dir.glob("*.json"):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            chunk_data = json.load(f)

                        chunk_data["project"] = project_dir.name
                        documents.append(chunk_data)

                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")

        print(f"Loaded {len(documents)} documents for BM25 indexing")

        # Build and save the index
        self.bm25_index.build_index(documents)

        # Create output directory if needed
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.bm25_index.save_index(save_path)

    def search(
        self, query: str, top_k: int = 10, metadata_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using hybrid approach: Vector + BM25 + RRF + Reranking.

        Args:
            query: Search query
            top_k: Final number of results to return
            metadata_fields: Metadata fields to include

        Returns:
            List of retrieved and reranked documents
        """
        print(f"Hybrid search: retrieving top {self.retrieval_k} from each system")

        # 1. Get semantic search results (no reranking)
        semantic_results = self.vector_retriever.search(
            query=query, top_k=self.retrieval_k, metadata_fields=metadata_fields
        )

        # 2. Get BM25 search results
        bm25_rankings = self.bm25_index.search(query, top_k=self.retrieval_k)

        # 3. Convert to consistent format for fusion
        semantic_rankings = [
            (doc["chunk_id"], doc["score"]) for doc in semantic_results
        ]

        print(
            f"Retrieved {len(semantic_rankings)} semantic results, {len(bm25_rankings)} BM25 results"
        )

        # 4. Apply Anthropic's weighted fusion
        combined_rankings = weighted_fusion(
            semantic_rankings, bm25_rankings, semantic_weight=self.semantic_weight
        )

        print(f"Fusion produced {len(combined_rankings)} unique results")

        # 5. Build document list for reranking
        fusion_results = []
        semantic_results_dict = {doc["chunk_id"]: doc for doc in semantic_results}

        for chunk_id, fusion_score in combined_rankings:
            if chunk_id in semantic_results_dict:
                # Use existing document data
                result = semantic_results_dict[chunk_id].copy()
                result["fusion_score"] = fusion_score
                result["retrieval_method"] = "hybrid"
                fusion_results.append(result)
            else:
                # Document found only by BM25 - load it
                doc = self._load_document_by_chunk_id(chunk_id, metadata_fields)
                if doc:
                    doc["fusion_score"] = fusion_score
                    doc["retrieval_method"] = "bm25_only"
                    fusion_results.append(doc)

        print(f"Prepared {len(fusion_results)} documents for reranking")

        # 6. Apply reranking to configurable number of candidates
        rerank_count = min(len(fusion_results), self.rerank_candidates)
        candidates_for_reranking = fusion_results[:rerank_count]

        print(f"Reranking top {len(candidates_for_reranking)} candidates")
        reranked_results = self.reranker.rerank(query, candidates_for_reranking)

        # 7. Return final top_k results
        return reranked_results[:top_k]

    def _load_document_by_chunk_id(
        self, chunk_id: str, metadata_fields: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Load a document by chunk ID from the file system."""
        chunks_dir = Path(self.vector_retriever.chunks_dir)
        project = self.vector_retriever.chunk_to_project.get(chunk_id, "unknown")

        if project == "unknown":
            return None

        file_path = chunks_dir / project / f"{chunk_id}.json"

        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)

                result = {
                    "chunk_id": chunk_id,
                    "score": 0.0,  # Will be set by reranker
                    "project": project,
                    "augmented_content": chunk_data.get("augmented_content", ""),
                }

                if metadata_fields:
                    metadata = chunk_data.get("metadata", {})
                    for field in metadata_fields:
                        if field in metadata:
                            result[field] = metadata[field]

                return result

            except Exception as e:
                print(f"Error loading document {chunk_id}: {e}")
                return None

        return None
