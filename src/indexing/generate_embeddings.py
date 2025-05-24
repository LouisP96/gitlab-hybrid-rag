#!/usr/bin/env python
"""
Generate embeddings for GitLab RAG system.

This script processes augmented chunks and creates FAISS vector index for similarity search.
It creates a single combined index across all projects for cross-project search capabilities.
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    handlers = []
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    return logging.getLogger("generate_embeddings")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings and build vector index for GitLab RAG"
    )

    # Input/output options
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing augmented chunks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save embeddings and indexes",
    )

    # Embedding options
    parser.add_argument(
        "--model",
        type=str,
        default="Alibaba-NLP/gte-multilingual-base",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for embeddings (default: auto-detect)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="File to save logs to"
    )

    args = parser.parse_args()

    # Setup logging
    global logger
    logger = setup_logging(args.log_level, args.log_file)

    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Record start time
    start_time = datetime.now()
    logger.info(f"Starting embedding generation at {start_time}")

    # Device selection
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model with optimizations
    logger.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True, device=device)

    # Try to enable xformers optimizations
    try:
        transformer_model = model._first_module().auto_model
        transformer_model.config.unpad_inputs = True
        transformer_model.half()  # Use float16 precision
        logger.info("Enabled optimizations: unpadding and half precision")
    except Exception as e:
        logger.warning(f"Could not enable all optimizations: {str(e)}")

    # Get embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Model loaded successfully. Embedding dimension: {embedding_dim}")

    # Initialize index, mappings and counters
    index = faiss.IndexFlatIP(embedding_dim)
    all_chunk_ids = []
    chunk_to_project = {}
    total_chunks = 0
    processed_chunks = 0
    error_chunks = 0

    # Process all projects
    all_projects = [d for d in input_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(all_projects)} projects to process")

    # Count total chunks first for progress tracking
    for project_dir in all_projects:
        json_files = list(project_dir.glob("*.json"))
        total_chunks += len(json_files)

    logger.info(f"Total chunks to process: {total_chunks}")

    # Process each project
    for project_dir in all_projects:
        project_name = project_dir.name
        logger.info(f"Processing project: {project_name}")

        # Get all JSON files
        json_files = list(project_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} chunks in project {project_name}")

        # Process in batches to manage memory
        batch_size = args.batch_size

        for i in range(0, len(json_files), batch_size):
            batch_end = min(i + batch_size, len(json_files))
            batch_files = json_files[i:batch_end]

            batch_texts = []
            batch_ids = []

            # Load each chunk in this batch
            for json_file in batch_files:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)

                    chunk_id = chunk_data.get("chunk_id", str(json_file.stem))
                    augmented_content = chunk_data.get("augmented_content", "")

                    if augmented_content:
                        batch_texts.append(augmented_content)
                        batch_ids.append(chunk_id)

                        # Store project mapping
                        chunk_to_project[chunk_id] = project_name
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {str(e)}")
                    error_chunks += 1

            # Generate embeddings for this batch
            if batch_texts:
                try:
                    with torch.no_grad():
                        batch_embeddings = model.encode(
                            batch_texts,
                            batch_size=256,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                        )

                    # Add to global index
                    index.add(batch_embeddings.astype(np.float32))

                    # Store ID mapping
                    all_chunk_ids.extend(batch_ids)

                    processed_chunks += len(batch_ids)

                    # Log progress
                    if processed_chunks % 1000 == 0 or processed_chunks == total_chunks:
                        logger.info(
                            f"Progress: {processed_chunks}/{total_chunks} chunks ({processed_chunks / total_chunks:.1%})"
                        )

                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {str(e)}")
                    error_chunks += len(batch_texts)

    # Save the combined index
    try:
        index_path = output_dir / "combined_index.faiss"
        logger.info(f"Saving combined index to {index_path}")
        faiss.write_index(index, str(index_path))

        # Save the unified ID mapping
        with open(output_dir / "combined_id_mapping.json", "w") as f:
            json.dump(all_chunk_ids, f)

        # Save the chunk to project mapping
        with open(output_dir / "chunk_to_project.json", "w") as f:
            json.dump(chunk_to_project, f)

        logger.info(
            f"Created combined vector index with {len(all_chunk_ids)} chunks from {len(all_projects)} projects"
        )
    except Exception as e:
        logger.error(f"Error saving index and mappings: {str(e)}")

    # Record end time and log statistics
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 50)
    logger.info("Embedding Generation Complete")
    logger.info(f"Started: {start_time}")
    logger.info(f"Finished: {end_time}")
    logger.info(f"Duration: {duration}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Successfully processed: {processed_chunks}")
    logger.info(f"Errors: {error_chunks}")
    logger.info(f"Index size: {len(all_chunk_ids)}")
    logger.info(f"Index dimension: {embedding_dim}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
