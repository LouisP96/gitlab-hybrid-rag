#!/usr/bin/env python3
"""
Metadata Augmentation Script for GitLab RAG

This script processes raw chunks by adding metadata prefixes to content,
preparing them for embedding.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any


def augment_chunk_with_metadata(chunk_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Augment chunk with its metadata.

    Args:
        chunk_data: Dictionary containing chunk data

    Returns:
        Dictionary with 'content' field replaced by augmented content
    """
    content = chunk_data.get("content", "")
    metadata = chunk_data.get("metadata", {})

    # initialise augmented content with a template
    metadata_fields = []

    # Add metadata fields if they exist
    if metadata.get("project"):
        metadata_fields.append(f"[PROJECT: {metadata['project']}]")

    if metadata.get("path"):
        metadata_fields.append(f"[PATH: {metadata['path']}]")

    if metadata.get("type"):
        metadata_fields.append(f"[TYPE: {metadata['type']}]")

    if metadata.get("language"):
        metadata_fields.append(f"[LANGUAGE: {metadata['language']}]")

    if metadata.get("name"):
        metadata_fields.append(f"[NAME: {metadata['name']}]")

    if metadata.get("created_at"):
        metadata_fields.append(f"[CREATED AT: {metadata['created_at']}]")

    if metadata.get("state"):
        metadata_fields.append(f"[STATE: {metadata['state']}]")

    # Create the augmented content by joining the metadata prefix with the content
    metadata_prefix = "\n".join(metadata_fields)
    if metadata_prefix:
        augmented_content = f"{metadata_prefix}\n\n{content}"
    else:
        augmented_content = content

    # Replace the content field with the augmented content
    result = chunk_data.copy()
    result["content"] = augmented_content

    return result


def process_chunks(input_dir: Path, output_dir: Path):
    """
    Process chunks by adding metadata prefixes and save them to output directory.

    Args:
        input_dir: Directory containing chunked documents
        output_dir: Directory to save augmented chunks to
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of projects
    projects = [d for d in input_dir.iterdir() if d.is_dir()]
    logging.info(f"Found {len(projects)} projects to process")

    # Process each project
    total_processed = 0
    total_errors = 0

    for project_dir in projects:
        project_name = project_dir.name

        # Create output directory for this project
        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(exist_ok=True)

        # Get all JSON files in this project
        json_files = list(project_dir.glob("*.json"))
        logging.info(f"Processing {len(json_files)} chunks from {project_name}")

        project_processed = 0
        project_errors = 0

        # Process chunks in batches to show periodic progress
        batch_size = 10000
        for i in range(0, len(json_files), batch_size):
            batch_end = min(i + batch_size, len(json_files))
            batch_files = json_files[i:batch_end]

            for json_file in batch_files:
                try:
                    # Load chunk data
                    with open(json_file, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)

                    # Augment chunk with metadata
                    augmented_chunk = augment_chunk_with_metadata(chunk_data)

                    # Save augmented chunk
                    output_file = project_output_dir / json_file.name
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(augmented_chunk, f, ensure_ascii=False, indent=2)

                    project_processed += 1

                except Exception as e:
                    logging.error(f"Error processing {json_file}: {str(e)}")
                    project_errors += 1

            # Log progress after each batch
            logging.info(
                f"Project {project_name}: Processed {project_processed}/{len(json_files)} chunks"
            )

        logging.info(
            f"Completed project {project_name}: {project_processed} processed, {project_errors} errors"
        )
        total_processed += project_processed
        total_errors += project_errors

    logging.info(
        f"Augmentation complete. Processed {total_processed} chunks with {total_errors} errors."
    )
    logging.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment chunks with metadata for GitLab RAG system"
    )

    # Input/output options
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/chunked_output",
        help="Directory containing chunked documents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/enriched_output",
        help="Directory to save augmented chunks to",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Process chunks
    process_chunks(input_dir, output_dir)


if __name__ == "__main__":
    main()
