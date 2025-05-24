#!/usr/bin/env python
"""
Chunking script for GitLab RAG system.

This script chunks processed documents with configurable parameters.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import pickle

from src.processing.base_processor import Document
from src.indexing.chunking import chunk_documents, Chunk


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    return logging.getLogger("chunk_data")


def load_documents(
    input_dir: Path, project_name: Optional[str] = None
) -> List[Document]:
    """
    Load processed documents from the input directory.

    Args:
        input_dir: Directory containing processed documents
        project_name: Optional project name filter

    Returns:
        List of Document objects
    """
    documents = []
    projects = []

    # Get list of projects
    if project_name:
        if (input_dir / project_name).exists():
            projects = [project_name]
        else:
            raise ValueError(f"Project directory not found: {project_name}")
    else:
        projects = [d.name for d in input_dir.iterdir() if d.is_dir()]

    logger.info(f"Loading documents from {len(projects)} projects")

    for project in projects:
        project_dir = input_dir / project

        # Find processor directories
        processor_dirs = [d for d in project_dir.iterdir() if d.is_dir()]

        for processor_dir in processor_dirs:
            # Load JSON documents from this processor
            json_files = list(processor_dir.glob("*.json"))

            logger.info(
                f"Loading {len(json_files)} documents from {processor_dir.name}"
            )

            for json_file in json_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        doc_data = json.load(f)

                    # Create Document object
                    doc = Document(
                        content=doc_data.get("content", ""),
                        metadata=doc_data.get("metadata", {}),
                        doc_id=doc_data.get("doc_id", str(json_file)),
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {str(e)}")

    logger.info(f"Loaded {len(documents)} documents total")
    return documents


def save_chunks(
    chunks: List[Chunk],
    output_dir: Path,
    save_format: str = "json",
    by_project: bool = True,
) -> None:
    """
    Save chunks to the output directory.

    Args:
        chunks: List of chunks to save
        output_dir: Directory to save chunks to
        save_format: Format to save chunks in (json or pickle)
        by_project: Whether to organize output by project
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if by_project:
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

            logger.info(f"Saving {len(project_chunks)} chunks for project {project}")

            if save_format == "json":
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
            else:  # pickle
                chunks_file = project_dir / "chunks.pkl"
                with open(chunks_file, "wb") as f:
                    pickle.dump(project_chunks, f)
    else:
        # Save all chunks to a single directory
        logger.info(f"Saving {len(chunks)} chunks to {output_dir}")

        if save_format == "json":
            for chunk in chunks:
                chunk_file = output_dir / f"{chunk.chunk_id}.json"
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
        else:  # pickle
            chunks_file = output_dir / "chunks.pkl"
            with open(chunks_file, "wb") as f:
                pickle.dump(chunks, f)


def main():
    parser = argparse.ArgumentParser(
        description="Chunk processed documents for GitLab RAG system"
    )

    # Input/output options
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing processed documents",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save chunks to"
    )
    parser.add_argument("--project", type=str, help="Process only a specific project")

    # Chunking parameters
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Number of characters to overlap between chunks",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=50,
        help="Minimum size of a chunk to be included",
    )
    parser.add_argument(
        "--max-chunk-size", type=int, default=1024, help="Maximum size of a chunk"
    )

    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "pickle"],
        default="json",
        help="Format to save chunks in",
    )
    parser.add_argument(
        "--flat", action="store_true", help="Don't organize output by project"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()
    print(args)

    # Set up logging
    global logger
    logger = setup_logging(args.log_level)

    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Load documents
    documents = load_documents(input_dir, args.project)

    if not documents:
        logger.error("No documents found to process")
        return

    # Chunk documents
    logger.info(
        f"Chunking {len(documents)} documents with max chunk_size={args.max_chunk_size}, "
        f"chunk_overlap={args.chunk_overlap}"
    )

    chunks = chunk_documents(
        documents,
        chunk_overlap=args.chunk_overlap,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
    )

    logger.info(f"Created {len(chunks)} chunks")

    # Save chunks
    save_chunks(chunks, output_dir, save_format=args.format, by_project=not args.flat)

    logger.info(f"Chunking complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
