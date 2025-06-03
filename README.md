# GitLab RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for GitLab codebases that enables intelligent querying of code, documentation, issues, merge requests, and project metadata using advanced hybrid search with query rewriting.

## Features

### **Document Processing & Chunking**

- **Specialised Code Processors**: Extracts functions, classes, and methods from Python (with Google/NumPy/reST docstrings), R (with roxygen2 documentation), Julia (with docstrings), RMarkdown, and generic function detection for 20+ other languages

- **Documentation Processing**: Handles README files, markdown documents, configuration files, package specifications, and project documentation with appropriate chunking strategies

- **GitLab Metadata Extraction**: Processes issues, merge requests, wiki pages, milestones, releases, and CI/CD pipelines with custom chunking that preserves context and hierarchical structure

- **Context Enrichment**: Every chunk is prefixed with structured metadata tags (`[PROJECT: name] [PATH: file/path] [TYPE: function] [LANGUAGE: python]`) to improve search accuracy and provide essential context for retrieval

### **Hybrid Search Pipeline**

1. **Query Rewriting**: Uses Claude (`claude-sonnet-4-20250514`) to intelligently rewrite queries based on conversation context
   - Expands follow-up questions with relevant context from previous interactions
   - Preserves technical terms, function names, and file paths for improved search accuracy

2. **Dual Retrieval**: Combines semantic vector search with keyword-based BM25 search
   - **Semantic Search**: Uses `Alibaba-NLP/gte-multilingual-base` for embedding generation and FAISS for vector similarity
   - **BM25 Search**: Code-optimised keyword retrieval with minimal preprocessing to preserve technical terms

3. **Weighted Fusion**: Results combined using **tunable Reciprocal Rank Fusion (RRF)**: `score = semantic_weight × (1/(rank+1)) + bm25_weight × (1/(rank+1))`

4. **Neural Reranking**: Final reranking with `BAAI/bge-reranker-v2-m3` cross-encoder for optimal relevance

5. **Response Generation**: Answer generation using Claude for context-aware responses
   - Source attribution with metadata, retrieval methods, and reranking scores

### **Interactive Web Interface**
- **Real-time Chat**: Instant question answering with markdown rendering
- **Source Exploration**: Expandable source details with full metadata and retrieval methods

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for embeddings)
- GitLab access token
- Anthropic API key

### Setup
Clone the repo, then:
```bash
pip install -e .
```

## Usage Guide

### 1. Download GitLab Data
```bash
python scripts/download_data.py \
    --gitlab-url https://your-gitlab.com \
    --access-token YOUR_TOKEN \
    --output-dir data/gitlab_data \
    --clone-repo
```

### 2. Process the Data
```bash
python scripts/process_data.py \
    --input-dir data/gitlab_data \
    --output-dir data/processed_output \
```

### 3. Create Chunks
```bash
python scripts/chunk_data.py \
    --input-dir data/processed_output \
    --output-dir data/chunked_output \
    --min-chunk-size 50 \
    --max-chunk-size 1024 \
    --chunk-overlap 50
```

### 4. Enrich with Metadata
```bash
python scripts/enrich_chunks.py \
    --input-dir data/chunked_output \
    --output-dir data/enriched_output
```

### 5. Build Search Indexes
```bash
# Generate vector embeddings
python scripts/generate_embeddings.py \
    --input-dir data/enriched_output \
    --output-dir data/embeddings_output \
    --batch-size 64

# Build BM25 index
python scripts/build_bm25_index.py \
    --input-dir data/enriched_output \
    --output-file data/embeddings_output/bm25_index.pkl
```

### 6. Start the Web Interface
```bash
export ANTHROPIC_API_KEY=your_api_key_here

python run_app.py \
    --port 5000 \
    --semantic-weight 0.8 \
    --retrieval-k 25 \
    --rerank-candidates 20 \
    --reranker-batch-size 32
```

Visit `http://localhost:5000` to start asking questions about your codebase!
