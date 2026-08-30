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

In enhanced mode, query expansion generates multiple search queries from the user's question, retrieves from each, deduplicates, and feeds the merged context to a tuned prompt.

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

To use a specific chunking strategy:

```bash
python3 ingest.py --strategy enhanced_query
```

Available strategies: `baseline`, `small_chunks`, `large_chunks`, `section_aware`, `metadata_enriched`, `enhanced_query`

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

## Evaluation

The project includes an evaluation framework to measure retrieval quality and answer correctness.

### Running evaluation

```bash
python3 eval.py
```

For enhanced mode evaluation (query expansion + tuned prompt + TOP_K=10):

```bash
python3 eval.py --enhanced
```

Save results to a specific file:

```bash
python3 eval.py --enhanced results/my_test.json
```

### Test questions

`test_questions.json` contains 10 test Q&A pairs across four categories:
- **factual** — specific facts (e.g., "How does the Freeze Spell work?")
- **strategy** — advice and recommendations (e.g., "When should you upgrade your Town Hall?")
- **specific_th** — Town Hall-level-specific questions (e.g., "What troops are recommended for TH9?")
- **comparison** — contrasting options

Each question includes expected keywords and expected source pages for scoring.

### Metrics

| Metric | What it measures |
|---|---|
| **Retrieval Hit Rate** | Did any expected page appear in the retrieved chunks? |
| **Retrieval Recall** | Fraction of expected pages found in the top-k results |
| **Keyword Recall** | Fraction of expected keywords found in the LLM's answer |

### Running the full experiment

Compare all chunking strategies side-by-side:

```bash
python3 experiment.py
```

This re-ingests with each strategy, runs eval, and prints a comparison table. Results are saved to `experiment_results/`.

### Experiment Results

| Strategy | Chunks | Ret Hit | Ret Recall | KW Recall |
|---|---|---|---|---|
| baseline | 212 | 90% | 66% | 51% |
| small_chunks | 385 | 80% | 49% | 52% |
| large_chunks | 173 | 90% | 55% | 52% |
| section_aware | 212 | 90% | 66% | 51% |
| metadata_enriched | 212 | 90% | 66% | 51% |
| **enhanced_query** | **212** | **100%** | **91%** | **62%** |

**`enhanced_query` is the recommended strategy.** It uses the same baseline chunking (1000-char, 100-char overlap) but improves the query side with:
- **Query expansion** — the LLM generates 2 alternative search queries, retrieves for all 3, and merges results
- **Higher TOP_K (10)** — retrieves more candidates for better coverage
- **Tuned prompt** — instructs the LLM to name specific troops, strategies, and TH levels instead of generalizing

## Chunking Strategies

| Strategy | Chunk Size | Overlap | Description |
|---|---|---|---|
| `baseline` | 1000 | 100 | Standard fixed-size chunking |
| `small_chunks` | 500 | 50 | Smaller, more focused chunks |
| `large_chunks` | 1500 | 200 | Larger chunks with more context |
| `section_aware` | 1000 | 100 | Splits on PDF section headers, chunks respect topic boundaries |
| `metadata_enriched` | 1000 | 100 | Baseline + extracted TH levels and topic tags in metadata |
| `enhanced_query` | 1000 | 100 | Baseline chunking + query expansion + tuned prompt + TOP_K=10 |

## Project Structure

```
RAG_COC/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── ingest.py                # PDF parsing, chunking, embedding, storage
├── query.py                 # Retrieval + generation logic (default & enhanced modes)
├── main.py                  # CLI entry point
├── eval.py                  # Evaluation script (retrieval + answer metrics)
├── experiment.py            # Runs all strategies and compares results
├── test_questions.json      # 10 test Q&A pairs for evaluation
├── clash_of_clans_guide.pdf # Source PDF(s)
├── chroma_db/               # Vector store (created by ingest.py)
├── experiment_results/      # Per-strategy evaluation results
└── venv/                    # Python virtual environment
```

## Configuration

Key parameters you can tune (in `ingest.py` and `query.py`):

| Parameter | File | Default | What it does |
|---|---|---|---|
| `CHUNK_SIZE` | `ingest.py` | 1000 | Max characters per chunk |
| `CHUNK_OVERLAP` | `ingest.py` | 100 | Overlap between chunks |
| `TOP_K` | `query.py` | 5 | Chunks retrieved per query (default mode) |
| `ENHANCED_TOP_K` | `query.py` | 10 | Chunks retrieved per query (enhanced mode) |
| `model` | `query.py` | `llama3.2` | Ollama model for generation |
| `temperature` | `query.py` | 0 | LLM creativity (0 = deterministic) |
