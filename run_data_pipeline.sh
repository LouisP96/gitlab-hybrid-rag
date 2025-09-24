#!/usr/bin/env bash
set -e

function message {
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    PLAIN='\033[0m'
    
    echo -e "\n${GREEN}*** ${RED}$1 ${GREEN}***${PLAIN}\n"
}

# Check if scripts directory exists
if [ ! -d "scripts" ]; then
    echo "Error: 'scripts' directory not found!"
    echo "Please run this script from the project root directory."
    exit 1
fi

message "Downloading data"
python -m pip scripts.download_data \
    --gitlab-url= # Insert gitlab url here "https://gitlab..."

message "Processing data"
python -m scripts.process_data

message "Chunking data"
python -m scripts.chunk_data

message "Enriching chunks"
python -m scripts.enrich_chunks

message "Generating embeddings"
python -m scripts.generate_embeddings

message "Building BM25 index"
python -m scripts.build_bm25_index

message "Pipeline completed successfully"