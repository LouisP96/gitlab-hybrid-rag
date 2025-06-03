#!/usr/bin/env python3
"""
Process GitLab backup data for RAG system.

This script runs the processing pipeline on GitLab backup data,
extracting and structuring documents from code, issues, wiki pages, etc.
"""

import logging
import argparse
import time
import sys

from src.processing.pipeline import ProcessingPipeline


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Process GitLab backup data for RAG system"
    )
    parser.add_argument(
        "--input-dir", default="data/gitlab_data", help="Path to GitLab data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_data",
        help="Path to output directory for processed documents",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--project", help="Process only this specific project (optional)"
    )
    parser.add_argument(
        "--max_projects",
        type=int,
        help="Maximum number of projects to process (optional)",
    )

    return parser.parse_args()


def main():
    """Main entry point of the script"""
    # Parse arguments
    args = parse_arguments()

    # Log configuration
    print("--- GitLab Processing Script ---")
    print(f"Args: {args}")

    if args.project:
        print(f"Processing only project: {args.project}")
    if args.max_projects:
        print(f"Processing at most {args.max_projects} projects")

    try:
        # Create pipeline
        pipeline = ProcessingPipeline(
            gitlab_backup_dir=args.input_dir,
            output_dir=args.output_dir,
            max_workers=args.workers,
            log_level=logging.INFO,
        )

        # Get list of projects
        projects = pipeline._get_projects()

        # Limit to specific project or max number of projects
        if args.project:
            if args.project in projects:
                projects = [args.project]
            else:
                print(f"Project '{args.project}' not found in GitLab backup directory")
                sys.exit(1)

        elif args.max_projects and len(projects) > args.max_projects:
            print(
                f"Limiting to {args.max_projects} projects out of {len(projects)} total"
            )
            projects = projects[: args.max_projects]

        print(f"Found {len(projects)} projects to process")

        # Process all projects or specific project
        if args.project:
            start_time = time.time()
            print(f"Processing project: {args.project}")
            result = pipeline.process_project(args.project)

            # Count documents
            doc_count = sum(len(docs) for processor, docs in result.items())
            print(f"Processed {doc_count} documents from {args.project}")

        else:
            start_time = time.time()
            print("Processing all projects")
            results = pipeline.process_all_projects(parallel=True)

            # Count documents
            total_docs = sum(
                len(docs)
                for project_results in results.values()
                for processor, docs in project_results.items()
            )
            print(f"Processed {total_docs} documents from {len(results)} projects")

        # Log total time
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
