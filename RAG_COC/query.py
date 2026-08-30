"""
Query Pipeline for Clash of Clans RAG

This module handles the retrieval and generation sides of RAG:
1. Takes a user question and finds the most relevant chunks from ChromaDB
2. Passes those chunks as context to Ollama's llama3.2 model
3. Returns a grounded answer with source citations

Supports two modes:
  default   — basic retrieval (top 5) with standard prompt
  enhanced  — query expansion + top 10 retrieval + tuned prompt

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
TOP_K = 5  # number of chunks to retrieve per query (default mode)
ENHANCED_TOP_K = 10  # retrieve more candidates in enhanced mode

SYSTEM_PROMPT = """You are a Clash of Clans expert assistant. Answer the user's question
using ONLY the context provided below. If the context doesn't contain enough information
to answer, say so — don't make things up.

Keep answers concise and practical. When relevant, mention specific troop names,
Town Hall levels, or strategy details.

Context:
{context}"""

# enhanced prompt instructs the LLM to be more specific and exhaustive
ENHANCED_SYSTEM_PROMPT = """You are a Clash of Clans expert assistant. Answer the user's question
using ONLY the context provided below. If the context doesn't contain enough information
to answer, say so — don't make things up.

Rules:
- Name every specific troop, spell, hero, defense, or strategy mentioned in the context
  that is relevant to the question. Do not generalize — use exact names.
- If the context mentions Town Hall levels, include them in your answer.
- If the context contains a list or table, reproduce the key items rather than summarizing.
- Be thorough — cover all relevant points from the context, not just the first match.
- Keep your answer well-structured and practical.

Context:
{context}"""

# query expansion prompt — generates alternative search queries
EXPANSION_PROMPT = """Given the user's question about Clash of Clans, generate 2 alternative
search queries that would help find relevant information. Return ONLY the queries, one per line.
Do not number them or add any other text.

Question: {question}"""


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


def deduplicate_docs(docs):
    """Remove duplicate chunks (same source + page + content)."""
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:100])
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def expand_query(question, llm):
    """
    Use the LLM to generate alternative search queries for better retrieval.
    Returns the original question plus 2 expanded variants.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("human", EXPANSION_PROMPT),
    ])
    chain = prompt | llm | StrOutputParser()
    expanded = chain.invoke({"question": question})
    queries = [question]
    for line in expanded.strip().split("\n"):
        line = line.strip()
        if line and len(line) > 5:
            queries.append(line)
    return queries[:3]  # original + up to 2 expansions


def build_chain(vectorstore, enhanced=False):
    """
    Build the RAG chain: retriever → prompt → LLM → parse output.

    In enhanced mode:
    - Uses a tuned prompt that instructs specificity
    - Retriever is configured for higher top-k (used by enhanced_query)
    - Query expansion happens in the query() function
    """
    top_k = ENHANCED_TOP_K if enhanced else TOP_K
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    llm = ChatOllama(model="llama3.2", temperature=0)

    system_prompt = ENHANCED_SYSTEM_PROMPT if enhanced else SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query(question, chain, retriever, enhanced=False):
    """
    Run a question through the RAG pipeline.
    Returns the LLM's answer and the source documents used.

    In enhanced mode, expands the query into multiple search queries,
    retrieves for each, and deduplicates before answering.
    """
    if enhanced:
        llm = ChatOllama(model="llama3.2", temperature=0)
        queries = expand_query(question, llm)

        # retrieve docs for all query variants and merge
        all_docs = []
        for q in queries:
            all_docs.extend(retriever.invoke(q))
        all_docs = deduplicate_docs(all_docs)

        # build context from merged docs and generate answer
        context = format_docs(all_docs)
        system_prompt = ENHANCED_SYSTEM_PROMPT
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        answer_chain = prompt | llm | StrOutputParser()
        answer = answer_chain.invoke({"context": context, "question": question})

        return answer, all_docs
    else:
        answer = chain.invoke(question)
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
