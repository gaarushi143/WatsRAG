"""
PDF Ingestion Pipeline for Clash of Clans RAG

This script reads all PDF files from the project directory, splits them into
smaller chunks, generates vector embeddings using Ollama, and stores everything
in a local ChromaDB database for later retrieval.

Supports multiple chunking strategies via --strategy flag:
  baseline         1000-char chunks, 100-char overlap (default)
  small_chunks     500-char chunks, 50-char overlap
  large_chunks     1500-char chunks, 200-char overlap
  section_aware    splits on PDF section headers, then sub-splits large sections
  metadata_enriched  baseline chunks + extracted TH levels and topic tags

Pipeline: PDFs → Extract Text → Split into Chunks → Embed → Store in ChromaDB
"""

import os
import re
import sys
import glob
import shutil
import fitz  # PyMuPDF — used to extract text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---

PDF_DIR = os.path.dirname(os.path.abspath(__file__))  # directory containing the PDFs
CHROMA_DIR = os.path.join(PDF_DIR, "chroma_db")       # where the vector store is saved
COLLECTION_NAME = "clash_of_clans"                     # name of the ChromaDB collection

STRATEGIES = {
    "baseline":          {"chunk_size": 1000, "chunk_overlap": 100},
    "small_chunks":      {"chunk_size": 500,  "chunk_overlap": 50},
    "large_chunks":      {"chunk_size": 1500, "chunk_overlap": 200},
    "section_aware":     {"chunk_size": 1000, "chunk_overlap": 100},
    "metadata_enriched": {"chunk_size": 1000, "chunk_overlap": 100},
    "enhanced_query":    {"chunk_size": 1000, "chunk_overlap": 100},  # same chunking as baseline; improvements are on the query side
}

# section headers found in the PDF (uppercase lines that mark topic boundaries)
SECTION_HEADERS = [
    "TOWN HALL LEVELS", "TROOPS", "DEFENSES", "SPELLS",
    "QUICK REFERENCE", "ATTACK STRATEGIES",
]


def load_pdfs(pdf_dir):
    """
    Read all PDFs in the given directory and extract text page-by-page.
    Each page becomes a document with metadata (source filename + page number).
    Pages with fewer than 50 characters are skipped (likely blank or image-only).
    """
    docs = []
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s)")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"  Processing: {filename}")
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            text = doc[page_num].get_text().strip()
            # skip pages with very little text (blank pages, image-only pages)
            if len(text) > 50:
                docs.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "page": page_num + 1,
                    },
                })
        doc.close()

    print(f"Extracted {len(docs)} pages with content")
    return docs


def chunk_baseline(docs, chunk_size, chunk_overlap):
    """Standard fixed-size chunking with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    **doc["metadata"],
                    "chunk": i,
                },
            })
    return chunks


def chunk_section_aware(docs, chunk_size, chunk_overlap):
    """
    Group pages by detected section headers, then sub-split each section.
    Chunks stay within topic boundaries instead of cutting mid-topic.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # detect which section each page belongs to
    current_section = "GENERAL"
    section_pages = []

    for doc in docs:
        text_upper = doc["text"].upper()
        for header in SECTION_HEADERS:
            if header in text_upper:
                current_section = header
                break
        section_pages.append((current_section, doc))

    chunks = []
    for section, doc in section_pages:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    **doc["metadata"],
                    "chunk": i,
                    "section": section.title(),
                },
            })
    return chunks


def extract_th_levels(text):
    """Extract Town Hall level mentions from text (e.g., TH9, Town Hall 12)."""
    patterns = [
        r'\bTH\s*(\d{1,2})\b',
        r'Town\s+Hall\s+(\d{1,2})',
    ]
    levels = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            levels.add(int(match.group(1)))
    return sorted(levels)


def extract_topic(text):
    """Detect the dominant topic of a chunk based on keyword frequency."""
    topic_keywords = {
        "troops": ["troop", "barbarian", "archer", "giant", "dragon", "wizard",
                    "pekka", "miner", "hog rider", "valkyrie", "golem", "witch",
                    "lava hound", "bowler", "balloon", "healer"],
        "spells": ["spell", "rage", "heal", "freeze", "lightning", "earthquake",
                    "poison", "haste", "clone", "bat", "invisibility"],
        "defenses": ["defense", "cannon", "archer tower", "mortar", "air defense",
                      "wizard tower", "x-bow", "inferno", "eagle", "scattershot",
                      "tesla"],
        "strategy": ["attack", "strategy", "funnel", "queen walk", "hybrid",
                      "lavaloon", "gowipe", "barch", "smash", "spam"],
        "base_building": ["base", "layout", "compartment", "wall", "trap"],
    }

    text_lower = text.lower()
    scores = {}
    for topic, keywords in topic_keywords.items():
        scores[topic] = sum(1 for kw in keywords if kw in text_lower)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def chunk_metadata_enriched(docs, chunk_size, chunk_overlap):
    """
    Same chunking as baseline, but enriches each chunk's metadata with:
    - town_halls: list of TH levels mentioned in the chunk
    - topic: detected topic category (troops, spells, defenses, strategy, etc.)
    """
    base_chunks = chunk_baseline(docs, chunk_size, chunk_overlap)

    for chunk in base_chunks:
        th_levels = extract_th_levels(chunk["text"])
        topic = extract_topic(chunk["text"])
        chunk["metadata"]["town_halls"] = ",".join(str(t) for t in th_levels) if th_levels else ""
        chunk["metadata"]["topic"] = topic

    return base_chunks


def chunk_documents(docs, strategy="baseline"):
    """
    Split documents into chunks using the specified strategy.
    """
    config = STRATEGIES[strategy]
    chunk_size = config["chunk_size"]
    chunk_overlap = config["chunk_overlap"]

    if strategy in ("baseline", "small_chunks", "large_chunks", "enhanced_query"):
        chunks = chunk_baseline(docs, chunk_size, chunk_overlap)
    elif strategy == "section_aware":
        chunks = chunk_section_aware(docs, chunk_size, chunk_overlap)
    elif strategy == "metadata_enriched":
        chunks = chunk_metadata_enriched(docs, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"Created {len(chunks)} chunks (strategy: {strategy})")
    return chunks


def clear_chroma():
    """Remove existing ChromaDB data to start fresh."""
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"Cleared existing vector store at {CHROMA_DIR}")


def store_embeddings(chunks, batch_size=50):
    """
    Generate vector embeddings for each chunk using Ollama's nomic-embed-text
    model (runs locally), then store the vectors + text + metadata in a
    persistent ChromaDB collection on disk.

    Embeds in batches to avoid overwhelming Ollama with large requests.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print("Generating embeddings and storing in ChromaDB...")

    # create the store with the first batch
    first_batch = min(batch_size, len(texts))
    vectorstore = Chroma.from_texts(
        texts=texts[:first_batch],
        metadatas=metadatas[:first_batch],
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    # add remaining batches
    for i in range(first_batch, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        vectorstore.add_texts(
            texts=texts[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"  Embedded {end}/{len(texts)} chunks...")

    print(f"Stored {len(texts)} chunks in {CHROMA_DIR}")
    return vectorstore


def main():
    """
    Run the full ingestion pipeline:
    1. Load and extract text from all PDFs
    2. Split text into overlapping chunks (using the chosen strategy)
    3. Embed and store chunks in ChromaDB

    Usage: python3 ingest.py [--strategy baseline|small_chunks|large_chunks|section_aware|metadata_enriched]
    """
    # parse --strategy flag
    strategy = "baseline"
    if "--strategy" in sys.argv:
        idx = sys.argv.index("--strategy")
        if idx + 1 < len(sys.argv):
            strategy = sys.argv[idx + 1]
            if strategy not in STRATEGIES:
                print(f"Unknown strategy: {strategy}")
                print(f"Available: {', '.join(STRATEGIES.keys())}")
                sys.exit(1)

    print(f"=== Clash of Clans RAG Ingestion (strategy: {strategy}) ===\n")

    clear_chroma()
    docs = load_pdfs(PDF_DIR)
    chunks = chunk_documents(docs, strategy)
    store_embeddings(chunks)

    print("\nDone! Vector store ready at:", CHROMA_DIR)


if __name__ == "__main__":
    main()
