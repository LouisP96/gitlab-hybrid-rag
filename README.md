# GitLab RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for GitLab codebases that enables intelligent querying of code, documentation, issues, merge requests, and project metadata using advanced hybrid search.

## Features & Technology

### **Intelligent Document Processing & Chunking**

**Custom Processors with Specialised Logic:**
- **Python**: Extracts functions, classes, methods with Google/NumPy/reST docstring parsing
- **R**: Function extraction with roxygen2 documentation (`#'` comments) and parameter parsing
- **Julia**: Functions, types, structs with triple-quoted docstring extraction
- **RMarkdown**: Separates code chunks from markdown, preserves YAML headers
- **Generic Code**: 20+ languages (C++, Java, Go, Rust, etc.) with intelligent function detection

**GitLab Metadata Processing:**
- **Issues/MRs**: Custom chunking preserves titles with content, processes discussions and comments
- **Wiki Pages**: Section-aware chunking maintains hierarchical structure with headers
- **Project Metadata**: Milestones, releases, CI/CD pipelines with structured metadata extraction

**Metadata Augmentation:**
- **Chunk Enrichment**: Every chunk prefixed with structured metadata tags
- **Format**: `[PROJECT: name] [PATH: file/path] [TYPE: function] [LANGUAGE: python] [NAME: func_name]`
- **Enhanced Retrieval**: Metadata prefixes dramatically improve search accuracy and context

### **Advanced Hybrid Search Pipeline**

**3-Stage Retrieval Process:**
1. **Dual Retrieval**: Combines semantic vector search with keyword-based BM25 search
   - **Semantic Search**: Uses `Alibaba-NLP/gte-multilingual-base` for embedding generation and FAISS for vector similarity
   - **BM25 Search**: Code-optimised keyword retrieval with technical term preservation and minimal stemming

2. **Weighted Fusion**: Results combined using **tunable Reciprocal Rank Fusion (RRF)**
   - Configurable semantic/BM25 weight balance (default: 80%/20%)
   - Anthropic's weighted fusion: `score = semantic_weight × (1/(rank+1)) + bm25_weight × (1/(rank+1))`

3. **Neural Reranking**: Final reranking with `BAAI/bge-reranker-base` cross-encoder for optimal relevance

### **Response Generation**
- **Claude Sonnet 4**: Uses Anthropic's `claude-sonnet-4-20250514` for intelligent responses
- **Context-Aware**: Maintains conversation history and understands follow-up questions
- **Source Attribution**: Detailed citations with metadata, retrieval methods, and reranking scores
- **Code-Optimised**: Specialised system prompts for programming concepts and project structure

### **Interactive Web Interface**
- **Real-time Chat**: Instant question answering with streaming responses
- **Hybrid Search Indicators**: Shows semantic vs BM25 contribution and reranking scores
- **Source Exploration**: Expandable source details with full metadata and original ranks
- **Debug Tools**: API testing, configuration display, and performance monitoring
- **Environment Adaptive**: Auto-detects Posit Workbench, proxy configurations

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for embeddings)
- GitLab access token
- Anthropic API key

### Dependencies
```bash
pip install -r requirements.txt
```

## Setup Guide

### 1. Download GitLab Data
```bash
python src/download_data/download_data.py \
    --gitlab_url https://your-gitlab.com \
    --access_token YOUR_TOKEN \
    --output_directory gitlab_data \
    --clone_repo
```

### 2. Process the Data
```bash
python src/processing/process_data.py \
    --gitlab-dir gitlab_data \
    --output-dir data/processed_output \
    --workers 4
```

### 3. Create Chunks
```bash
python src/indexing/chunk_data.py \
    --input-dir data/processed_output \
    --output-dir data/chunked_output \
    --min-chunk_size 50 \
    --max-chunk-size 1024 \
    --chunk-overlap 50
```

### 4. Enrich with Metadata
```bash
python src/indexing/enrich_chunks.py \
    --input-dir data/chunked_output \
    --output-dir data/enriched_output
```

### 5. Build Search Indexes
```bash
# Generate vector embeddings
python src/indexing/generate_embeddings.py \
    --input-dir data/enriched_output \
    --output-dir data/embeddings_output

# Build BM25 index
python src/retrieval/build_bm25_index.py \
    --chunks-dir data/enriched_output \
    --output data/embeddings_output/bm25_index.pkl
```

### 6. Start the Web Interface
```bash
export ANTHROPIC_API_KEY=your_api_key_here

python run_app.py \
    --port 5000 \
    --semantic-weight 0.8 \
    --retrieval-k 50 \
    --rerank-candidates 30
```

Visit `http://localhost:5000` to start asking questions about your codebase!
