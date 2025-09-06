#!/usr/bin/env python3
"""
Generate embeddings for GitLab RAG system.

This script processes augmented chunks and creates FAISS vector index for similarity search.
It creates a single combined index across all projects for cross-project search capabilities.
"""

import json
import argparse
import logging
import torch
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime

def optimise_model(model):
    """Apply optimisations to the model."""
    transformer_model = model._first_module().auto_model
    transformer_model.config.unpad_inputs = True
    transformer_model.half()
    logging.info("Applied model optimisations: unpadding and half precision")


def load_chunk_data(json_files, project_name):
    """Load and yield chunk data from JSON files."""
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
        
        chunk_id = chunk_data.get("chunk_id", json_file.stem)
        content = chunk_data.get("content", "")
        
        if content:
            yield chunk_id, content, project_name


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings and build vector index for GitLab RAG"
    )

    # Input/output options
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/enriched_output",
        help="Directory containing augmented chunks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings_output",
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
        help="Device to use for embeddings",
    )

    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    start_time = datetime.now()
    logging.info(f"Starting embedding generation at {start_time}")

    # Device selection
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model with optimisations
    logging.info(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, trust_remote_code=True, device=device)
    optimise_model(model)

    # Get embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    logging.info(f"Model loaded successfully. Embedding dimension: {embedding_dim}")

    # initialise index and mappings
    index = faiss.IndexFlatIP(embedding_dim)
    all_chunk_ids = []
    chunk_to_project = {}
    processed_chunks = 0

    # Get all projects and count total chunks
    all_projects = [d for d in input_dir.iterdir() if d.is_dir()]
    total_chunks = sum(len(list(proj.glob("*.json"))) for proj in all_projects)
    logging.info(f"Found {len(all_projects)} projects with {total_chunks} total chunks")

    # Process all chunks
    batch_texts = []
    batch_ids = []
    
    for project_dir in all_projects:
        project_name = project_dir.name
        json_files = list(project_dir.glob("*.json"))
        logging.info(f"Processing {len(json_files)} chunks from {project_name}")
        
        for chunk_id, content, project in load_chunk_data(json_files, project_name):
            batch_texts.append(content)
            batch_ids.append(chunk_id)
            chunk_to_project[chunk_id] = project
            
            # Process batch when full
            if len(batch_texts) >= args.batch_size:
                with torch.no_grad():
                    embeddings = model.encode(
                        batch_texts,
                        batch_size=args.batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                
                index.add(embeddings.astype(np.float32))
                all_chunk_ids.extend(batch_ids)
                processed_chunks += len(batch_ids)
                
                # Reset batch
                batch_texts = []
                batch_ids = []
    
    # Process remaining batch
    if batch_texts:
        with torch.no_grad():
            embeddings = model.encode(
                batch_texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        index.add(embeddings.astype(np.float32))
        all_chunk_ids.extend(batch_ids)
        processed_chunks += len(batch_ids)

    # Save outputs
    faiss.write_index(index, str(output_dir / "combined_index.faiss"))
    (output_dir / "combined_id_mapping.json").write_text(json.dumps(all_chunk_ids))
    (output_dir / "chunk_to_project.json").write_text(json.dumps(chunk_to_project))

    # Summary
    duration = datetime.now() - start_time
    logging.info(f"Completed in {duration}. Processed {processed_chunks}/{total_chunks} chunks")


if __name__ == "__main__":
    main()