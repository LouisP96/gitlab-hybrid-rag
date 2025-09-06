#!/usr/bin/env python3
"""
Simple script that processes code, metadata, and documentation files
into structured documents for retrieval.
"""

import argparse
import sys
from pathlib import Path

from src.processing.processing_pipeline import ProcessingPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", default="data/gitlab_data", help="Path to GitLab data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_data",
        help="Path to output directory for processed documents",
    )
    parser.add_argument(
        "--project", help="Process only this specific project (optional)"
    )
    parser.add_argument(
        "--max-file-size",
        type=float,
        default=5.0,
        help="Maximum file size in MB to process (default: 5.0)",
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    try:
        # Create pipeline
        pipeline = ProcessingPipeline(input_dir, output_dir, args.max_file_size)

        if args.project:
            print(f"Processing single project: {args.project}")
            results = pipeline.process_project(args.project)

            total_docs = sum(len(docs) for docs in results.values())
            print(f"Processed {total_docs} documents from {args.project}")

        else:
            print("Processing all projects...")
            results = pipeline.process_all_projects()

            total_docs = sum(
                len(docs)
                for project_results in results.values()
                for docs in project_results.values()
            )
            print(f"Processed {total_docs} documents from {len(results)} projects")

        print("Processing completed successfully!")

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
