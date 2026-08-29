# Clash of Clans RAG

A Retrieval-Augmented Generation (RAG) system that answers Clash of Clans questions using PDF guides as its knowledge base. Everything runs locally — no API keys, no cloud services, no costs.

## Architecture

```
PDFs → Parse → Chunk → Embed → ChromaDB (vector store)
                                       ↓
User Question → Embed → Similarity Search → Top-K Chunks
                                                    ↓
                              Chunks + Question → Ollama LLM → Answer
```

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| PDF parsing | PyMuPDF (fitz) | Extract text from PDF files |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Split text into overlapping chunks |
| Embeddings | Ollama `nomic-embed-text` | Convert text to vectors (local) |
| Vector store | ChromaDB | Store and search embeddings (persistent, local) |
| LLM | Ollama `llama3.2` (3B) | Generate answers from retrieved context |
| Orchestration | LangChain + `langchain-ollama` | Wire all the pieces together |
| Interface | CLI | Terminal-based Q&A loop |

## Prerequisites

- **Python 3.9+**
- **Ollama** — install from [ollama.com](https://ollama.com)

## Setup

### 1. Install Ollama models

Make sure Ollama is running, then pull the two required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 2. Create virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your PDFs

Place your Clash of Clans PDF guides in the project root directory (alongside `ingest.py`).

### 4. Run ingestion

This parses all PDFs, splits them into chunks, generates embeddings, and stores them in ChromaDB. You only need to run this once (or again if you add/change PDFs).

```bash
python3 ingest.py
```

### 5. Run the app

```bash
python3 main.py
```

Type your question and press Enter. The system retrieves relevant chunks from the PDFs and generates an answer with source citations. Type `quit` or `exit` to stop.

### Example

```
You: How does the freeze spell work?

Thinking...

The freeze spell freezes ground and air troops, along with enemy defenses,
within a small radius of 3.5 tiles. Frozen units cannot move, attack, or heal.
The freeze spell has a brewing time of 3 minutes and requires a level 4 Spell Factory.

📚 Sources:
  - clash_of_clans_guide.pdf, page 38
  - clash_of_clans_guide.pdf, page 43
  - clash_of_clans_guide.pdf, page 47
```

## Project Structure

```
RAG_COC/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── ingest.py              # PDF parsing, chunking, embedding, storage
├── query.py               # Retrieval + generation logic
├── main.py                # CLI entry point
├── clash_of_clans_guide.pdf   # Source PDF(s)
├── chroma_db/             # Vector store (created by ingest.py)
└── venv/                  # Python virtual environment
```

## Implementation Plan

The project was built in 4 steps:

1. **Project Setup** — virtual environment, dependencies, Ollama models
2. **PDF Ingestion** (`ingest.py`) — parse PDFs with PyMuPDF, split into ~1000-char chunks with 100-char overlap, embed with `nomic-embed-text`, store in persistent ChromaDB
3. **Query Pipeline** (`query.py`) — load ChromaDB, retrieve top-5 similar chunks per question, format them as context, send to `llama3.2` via LangChain, return answer with source citations
4. **CLI Interface** (`main.py`) — simple input loop that ties ingestion and querying together

## Configuration

Key parameters you can tune (in `ingest.py` and `query.py`):

| Parameter | File | Default | What it does |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | 1000 | Max characters per chunk |
| `CHUNK_OVERLAP` | `ingest.py` | 100 | Overlap between chunks |
| `TOP_K` | `query.py` | 5 | Number of chunks retrieved per query |
| `model` | `query.py` | `llama3.2` | Ollama model for generation |
| `temperature` | `query.py` | 0 | LLM creativity (0 = deterministic) |
