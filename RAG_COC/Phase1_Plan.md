# RAG System for Clash of Clans — Implementation Plan

## Context
Build a Retrieval-Augmented Generation (RAG) system over 12 Clash of Clans PDF guides. The system lets a user ask questions in the terminal and get answers grounded in the source material. Everything runs locally — no API costs.

## Architecture

```
PDFs → Parse → Chunk → Embed → ChromaDB (vector store)
                                       ↓
User Question → Embed → Similarity Search → Top-K Chunks
                                                    ↓
                              Chunks + Question → Ollama LLM → Answer
```

## Stack

| Layer | Tool |
|---|---|
| PDF parsing | `PyMuPDF` (fitz) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | Ollama `nomic-embed-text` (local, no API key) |
| Vector store | ChromaDB (persistent, local) |
| LLM | Ollama `llama3.2` (3B) |
| Orchestration | LangChain + `langchain-ollama` |
| Interface | CLI (Python `input()` loop) |

## Implementation Steps

### Step 1: Project Setup
- Create `requirements.txt` with dependencies: `langchain`, `langchain-community`, `langchain-ollama`, `chromadb`, `pymupdf`
- Set up Python virtual environment
- Ensure Ollama is installed and running, then pull models:
  - `ollama pull llama3.2`
  - `ollama pull nomic-embed-text`

### Step 2: PDF Ingestion (`ingest.py`)
- Read all PDFs from the project directory using PyMuPDF
- Extract text page-by-page, preserving source filename + page number as metadata
- Split text into ~500-token chunks with ~50-token overlap using `RecursiveCharacterTextSplitter`
- Generate embeddings via Ollama's `nomic-embed-text`
- Store chunks + embeddings + metadata in a persistent ChromaDB collection (`./chroma_db/`)

### Step 3: Query Pipeline (`query.py`)
- Load the persisted ChromaDB collection
- Accept user question from CLI
- Embed the question using the same `nomic-embed-text` model
- Retrieve top-5 most similar chunks
- Build a prompt: system instruction + retrieved chunks + user question
- Send to Ollama LLM for answer generation
- Print the answer with source citations (filename + page)

### Step 4: CLI Interface (`main.py`)
- Simple `while True` loop: prompt → retrieve → generate → print
- Show sources alongside answers
- `quit` / `exit` to stop

### Step 5: Testing & Iteration
- Test with queries like:
  - "What's the best attack strategy for Town Hall 11?"
  - "How does the freeze spell work?"
  - "When should I upgrade my town hall?"
  - "What are village guards?"
- Tune chunk size, overlap, and top-k if answers are weak

## Files to Create
- `requirements.txt` — dependencies
- `ingest.py` — PDF parsing, chunking, embedding, storage
- `query.py` — retrieval + generation logic
- `main.py` — CLI entry point

## Verification
1. Run `python ingest.py` — should process all 12 PDFs and create `./chroma_db/`
2. Run `python main.py` — ask a question, get a sourced answer
3. Verify answers reference the correct PDFs and are factually grounded in the content
