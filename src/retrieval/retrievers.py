import numpy as np
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from .bm25 import BM25Index, weighted_fusion


class RAGRetriever:
    def __init__(
        self,
        index_path="data/embeddings_output/combined_index.faiss",
        id_mapping_path="data/embeddings_output/combined_id_mapping.json",
        project_mapping_path="data/embeddings_output/chunk_to_project.json",
        chunks_dir="data/enriched_output",
        model_name="Alibaba-NLP/gte-multilingual-base",
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

        # Load model with optimisations
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        transformer_model = self.model._first_module().auto_model
        transformer_model.config.unpad_inputs = True
        transformer_model.half()

    def search(
        self,
        query,
        retrieval_top_k=5,
    ):
        """
        Search for relevant chunks.

        Args:
            query: The query text to search for
            retrieval_top_k: Number of results to return

        Returns:
            List of chunk results with metadata
        """

        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Search the index
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), retrieval_top_k
        )

        # Process results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.id_mapping):
                chunk_id = self.id_mapping[idx]
                project = self.chunk_to_project.get(chunk_id, "unknown")

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
                        "content": chunk_data.get("content", ""),
                    }

                    # Add all metadata fields if they exist
                    metadata = chunk_data.get("metadata", {})
                    result.update(metadata)

                    # Set retrieval method for semantic-only search
                    result["retrieval_method"] = "semantic"

                    # Add result to list
                    results.append(result)

        # Check if we have fewer results than requested
        if len(results) < retrieval_top_k:
            print(
                f"Warning: RAGRetriever requested {retrieval_top_k} results but only {len(results)} results found"
            )

        # Return retrieval_top_k results
        return results[:retrieval_top_k]


class HybridRetriever(RAGRetriever):
    """
    Hybrid retriever that combines semantic search and BM25.
    Inherits from RAGRetriever and extends it with BM25 + fusion capabilities.
    """

    def __init__(
        self,
        bm25_index_path: str = "data/embeddings_output/bm25_index.pkl",
        semantic_weight: float = 0.8,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        per_system_k: int = 25,  # How many to retrieve from each system
        # RAGRetriever parameters
        index_path="data/embeddings_output/combined_index.faiss",
        id_mapping_path="data/embeddings_output/combined_id_mapping.json",
        project_mapping_path="data/embeddings_output/chunk_to_project.json",
        chunks_dir="data/enriched_output",
        model_name="Alibaba-NLP/gte-multilingual-base",
    ):
        """
        initialise hybrid retriever.

        Args:
            bm25_index_path: Path to BM25 index
            semantic_weight: Weight for semantic search in fusion (default 0.8)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            per_system_k: Number of results to retrieve from each system (default 25)
            # RAGRetriever parameters passed to parent
            index_path: Path to FAISS index
            id_mapping_path: Path to ID mapping file
            project_mapping_path: Path to chunk-to-project mapping
            chunks_dir: Directory containing chunk files
            model_name: Sentence transformer model name
        """
        # Initialise parent RAGRetriever
        super().__init__(
            index_path=index_path,
            id_mapping_path=id_mapping_path,
            project_mapping_path=project_mapping_path,
            chunks_dir=chunks_dir,
            model_name=model_name,
        )

        # Hybrid-specific parameters
        self.semantic_weight = semantic_weight
        self.per_system_k = per_system_k

        # Initialise BM25 index
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)

        # Load or build BM25 index
        if Path(bm25_index_path).exists():
            print(f"Loading BM25 index from {bm25_index_path}")
            self.bm25_index.load_index(bm25_index_path)
        else:
            raise FileNotFoundError(f"BM25 index not found at {bm25_index_path}. ")

    def search(self, query: str, retrieval_top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using hybrid approach: Vector + BM25 + RRF.

        Args:
            query: Search query
            retrieval_top_k: Final number of results to return

        Returns:
            List of retrieved documents with fusion scores
        """
        print(f"Hybrid search: retrieving top {self.per_system_k} from each system")

        # 1. Get semantic search results using parent's method
        semantic_results = super().search(
            query=query,
            retrieval_top_k=self.per_system_k,
        )

        # 2. Get BM25 search results
        bm25_rankings = self.bm25_index.search(query, per_system_k=self.per_system_k)

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

        # Check if we have fewer results than requested
        if len(combined_rankings) < retrieval_top_k:
            print(
                f"Warning: HybridRetriever requested {retrieval_top_k} results but only {len(combined_rankings)} unique results available after fusion"
            )

        # 5. Load documents inline and build results

        results = []
        for chunk_id, fusion_score in combined_rankings[:retrieval_top_k]:
            project = self.chunk_to_project.get(chunk_id, "unknown")
            if project == "unknown":
                continue

            # Load document from file
            file_path = self.chunks_dir / project / f"{chunk_id}.json"
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)

                    # Build result with fusion score
                    result = {
                        "chunk_id": chunk_id,
                        "score": fusion_score,
                        "project": project,
                        "content": chunk_data.get("content", ""),
                    }

                    # Add all metadata fields
                    metadata = chunk_data.get("metadata", {})
                    result.update(metadata)

                    # Determine retrieval method
                    was_in_semantic = chunk_id in [r[0] for r in semantic_rankings]
                    was_in_bm25 = chunk_id in [r[0] for r in bm25_rankings]

                    if was_in_semantic and was_in_bm25:
                        result["retrieval_method"] = "hybrid"
                    elif was_in_semantic:
                        result["retrieval_method"] = "semantic_only"
                    else:
                        result["retrieval_method"] = "bm25_only"

                    results.append(result)

                except Exception as e:
                    print(f"Error loading document {chunk_id}: {e}")

        return results
