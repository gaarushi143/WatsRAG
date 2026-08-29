"""
CLI Interface for Clash of Clans RAG

Simple terminal loop: type a question, get an answer grounded in the
Clash of Clans PDF guides, with source citations. Type 'quit' or 'exit' to stop.

Requires: run ingest.py first to populate the ChromaDB vector store.
"""

from query import load_vectorstore, build_chain, query, print_result


def main():
    print("=== Clash of Clans RAG ===")
    print("Ask anything about Clash of Clans. Type 'quit' or 'exit' to stop.\n")

    print("Loading vector store...")
    vectorstore = load_vectorstore()
    chain, retriever = build_chain(vectorstore)
    print("Ready!\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print("\nThinking...")
        answer, docs = query(question, chain, retriever)
        print_result(answer, docs)
        print()


if __name__ == "__main__":
    main()
