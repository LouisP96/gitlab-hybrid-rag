import json
import nltk
from pathlib import Path
from query_embeddings import BM25Index

# Download required NLTK data (run once)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def build_bm25_index():
    """CLI script to pre-build BM25 index."""
    import argparse

    parser = argparse.ArgumentParser(description="Build BM25 index for hybrid search")
    parser.add_argument(
        "--chunks-dir",
        default="data/enriched_output",
        help="Directory containing chunk files",
    )
    parser.add_argument(
        "--output",
        default="data/embeddings_output/bm25_index.pkl",
        help="Output path for BM25 index",
    )
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")

    args = parser.parse_args()

    # Initialize BM25 index
    bm25_index = BM25Index(k1=args.k1, b=args.b)

    # Load documents
    chunks_dir = Path(args.chunks_dir)
    documents = []

    print("Loading documents...")
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

    print(f"Loaded {len(documents)} documents")

    # Build and save index
    bm25_index.build_index(documents)

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bm25_index.save_index(args.output)

    print(f"BM25 index saved to {args.output}")


if __name__ == "__main__":
    build_bm25_index()
