"""
PDF Ingestion Pipeline for Clash of Clans RAG

This script reads all PDF files from the project directory, splits them into
smaller chunks, generates vector embeddings using Ollama, and stores everything
in a local ChromaDB database for later retrieval.

Pipeline: PDFs → Extract Text → Split into Chunks → Embed → Store in ChromaDB
"""

import os
import glob
import fitz  # PyMuPDF — used to extract text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---

PDF_DIR = os.path.dirname(os.path.abspath(__file__))  # directory containing the PDFs
CHROMA_DIR = os.path.join(PDF_DIR, "chroma_db")       # where the vector store is saved
COLLECTION_NAME = "clash_of_clans"                     # name of the ChromaDB collection

CHUNK_SIZE = 1000    # max characters per chunk
CHUNK_OVERLAP = 100  # overlapping chars between consecutive chunks to preserve context


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


def chunk_documents(docs):
    """
    Split each page's text into smaller, overlapping chunks.
    Smaller chunks improve retrieval accuracy because the embedding
    captures a more focused topic. Overlap ensures we don't lose
    context at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # try splitting at paragraph breaks first, then sentences, then words
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    **doc["metadata"],  # carry over source filename and page number
                    "chunk": i,         # chunk index within this page
                },
            })

    print(f"Created {len(chunks)} chunks")
    return chunks


def store_embeddings(chunks):
    """
    Generate vector embeddings for each chunk using Ollama's nomic-embed-text
    model (runs locally), then store the vectors + text + metadata in a
    persistent ChromaDB collection on disk.
    """
    # this embedding model runs locally via Ollama — no API key needed
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print("Generating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,       # saves to disk so we don't re-embed every time
        collection_name=COLLECTION_NAME,
    )
    print(f"Stored {len(texts)} chunks in {CHROMA_DIR}")
    return vectorstore


def main():
    """
    Run the full ingestion pipeline:
    1. Load and extract text from all PDFs
    2. Split text into overlapping chunks
    3. Embed and store chunks in ChromaDB
    """
    print("=== Clash of Clans RAG Ingestion ===\n")

    docs = load_pdfs(PDF_DIR)
    chunks = chunk_documents(docs)
    store_embeddings(chunks)

    print("\nDone! Vector store ready at:", CHROMA_DIR)


if __name__ == "__main__":
    main()
