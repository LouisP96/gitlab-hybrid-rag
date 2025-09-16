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
python scripts/download_data.py \
    --gitlab-url= # Insert gitlab url here "https://gitlab..."

message "Processing data"
python scripts/process_data.py

message "Chunking data"
python scripts/chunk_data.py

message "Enriching chunks"
python scripts/enrich_chunks.py

message "Generating embeddings"
python scripts/generate_embeddings.py

message "Building BM25 index"
python scripts/build_bm25_index.py

message "Pipeline completed successfully"