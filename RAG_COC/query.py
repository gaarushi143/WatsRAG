"""
Query Pipeline for Clash of Clans RAG

This module handles the retrieval and generation sides of RAG:
1. Takes a user question and finds the most relevant chunks from ChromaDB
2. Passes those chunks as context to Ollama's llama3.2 model
3. Returns a grounded answer with source citations

Used by main.py — can also be imported and called directly.
"""

import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "clash_of_clans"
TOP_K = 5  # number of chunks to retrieve per query

SYSTEM_PROMPT = """You are a Clash of Clans expert assistant. Answer the user's question
using ONLY the context provided below. If the context doesn't contain enough information
to answer, say so — don't make things up.

Keep answers concise and practical. When relevant, mention specific troop names,
Town Hall levels, or strategy details.

Context:
{context}"""


def load_vectorstore():
    """
    Load the persisted ChromaDB vector store from disk.
    Uses the same embedding model (nomic-embed-text) that was used during ingestion
    so that query embeddings are compatible with stored embeddings.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def format_docs(docs):
    """
    Format retrieved documents into a single string for the LLM prompt.
    Each chunk is separated by a divider for clarity.
    """
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def build_chain(vectorstore):
    """
    Build the RAG chain: retriever → prompt → LLM → parse output.

    The retriever finds the top-k most similar chunks to the user's question.
    Those chunks are injected into the prompt as context, and the LLM generates
    an answer grounded in that context.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # llama3.2 (3B) — lightweight local model via Ollama
    llm = ChatOllama(model="llama3.2", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # LangChain Expression Language (LCEL) chain:
    # 1. Retrieve relevant chunks and pass the question through
    # 2. Format chunks into context string
    # 3. Fill the prompt template
    # 4. Send to LLM
    # 5. Parse the output as a plain string
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query(question, chain, retriever):
    """
    Run a question through the RAG pipeline.
    Returns the LLM's answer and the source documents used.
    """
    # get the answer from the chain
    answer = chain.invoke(question)

    # separately retrieve the docs so we can show citations
    source_docs = retriever.invoke(question)

    return answer, source_docs


def print_result(answer, source_docs):
    """
    Display the answer and list the source documents that were used.
    """
    print(f"\n{answer}")

    # deduplicate sources (same file + page might appear from multiple chunks)
    seen = set()
    sources = []
    for doc in source_docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append(key)

    print("\n📚 Sources:")
    for source, page in sources:
        print(f"  - {source}, page {page}")
