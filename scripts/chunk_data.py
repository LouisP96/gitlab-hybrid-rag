#!/usr/bin/env python3
"""
Chunking script for GitLab RAG system.
Chunks processed documents into smaller pieces for retrieval.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List

from src.processing.document import Document
from src.indexing.chunking import chunk_documents


def load_documents(input_dir: Path) -> List[Document]:
    """Load processed documents from the input directory."""
    documents = []

    # Get list of projects
    projects = [d.name for d in input_dir.iterdir() if d.is_dir()]

    logging.info(f"Loading documents from {len(projects)} projects")

    for project in projects:
        project_dir = input_dir / project

        # Load from known processor directories
        for processor_type in ["code", "documentation", "metadata"]:
            processor_dir = project_dir / processor_type
            if not processor_dir.exists():
                continue

            json_files = list(processor_dir.glob("*.json"))
            logging.info(
                f"Loading {len(json_files)} {processor_type} documents from {project}"
            )

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        doc_data = json.load(f)

                    doc = Document(
                        content=doc_data.get("content", ""),
                        metadata=doc_data.get("metadata", {}),
                        doc_id=doc_data.get("doc_id", str(json_file.name)),
                    )
                    documents.append(doc)
                except Exception as e:
                    logging.error(f"Error loading {json_file}: {e}")

    logging.info(f"Loaded {len(documents)} documents total")
    return documents


def save_chunks(chunks, output_dir: Path):
    """Save chunks organized by project."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group chunks by project
    chunks_by_project = {}
    for chunk in chunks:
        project = chunk.metadata.get("project", "unknown")
        if project not in chunks_by_project:
            chunks_by_project[project] = []
        chunks_by_project[project].append(chunk)

    # Save each project's chunks
    for project, project_chunks in chunks_by_project.items():
        project_dir = output_dir / project
        project_dir.mkdir(exist_ok=True)

        logging.info(f"Saving {len(project_chunks)} chunks for {project}")

        for chunk in project_chunks:
            chunk_file = project_dir / f"{chunk.chunk_id}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "chunk_id": chunk.chunk_id,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


def main():
    parser = argparse.ArgumentParser(description="Chunk processed documents for RAG")
    parser.add_argument(
        "--input-dir",
        default="data/processed_data",
        help="Directory containing processed documents",
    )
    parser.add_argument(
        "--output-dir",
        default="data/chunked_output",
        help="Directory to save chunks to",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=1024,
        help="Maximum chunk size",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap size",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=50,
        help="Minimum chunk size",
    )

    args = parser.parse_args()

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load documents
    documents = load_documents(input_dir)
    if not documents:
        logging.error("No documents found to process")
        return

    # Chunk documents
    logging.info(
        f"Chunking {len(documents)} documents (max_chunk_size={args.max_chunk_size})"
    )
    chunks = chunk_documents(
        documents,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
    )

    logging.info(f"Created {len(chunks)} chunks")

    # Save chunks
    save_chunks(chunks, output_dir)
    logging.info(f"Chunking complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
