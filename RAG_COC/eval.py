"""
Evaluation Script for Clash of Clans RAG

Measures retrieval quality and answer correctness against a set of
test questions with known expected answers and source pages.

Metrics:
  - retrieval_hit: did any expected page appear in the top-k results?
  - retrieval_recall: fraction of expected pages found in top-k
  - keyword_recall: fraction of expected keywords found in the LLM answer

Usage:
  python3 eval.py [results_file]              # default mode
  python3 eval.py --enhanced [results_file]   # enhanced query mode
"""

import json
import os
import sys
import time
from query import load_vectorstore, build_chain, query

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(PROJECT_DIR, "test_questions.json")
DEFAULT_RESULTS_FILE = os.path.join(PROJECT_DIR, "eval_results.json")


def load_test_questions(path=TEST_FILE):
    with open(path) as f:
        return json.load(f)


def eval_retrieval(source_docs, expected_pages):
    """
    Measure how well the retriever found the right pages.
    Returns (hit, recall, retrieved_pages).
    """
    retrieved_pages = [doc.metadata.get("page") for doc in source_docs]

    # did we find ANY of the expected pages?
    found = [p for p in expected_pages if p in retrieved_pages]
    hit = 1 if len(found) > 0 else 0

    # what fraction of expected pages did we find?
    recall = len(found) / len(expected_pages) if expected_pages else 0

    return hit, recall, retrieved_pages


def eval_answer(answer, expected_keywords):
    """
    Measure how many expected keywords appear in the answer (case-insensitive).
    Returns (keyword_recall, found_keywords, missing_keywords).
    """
    answer_lower = answer.lower()
    found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    recall = len(found) / len(expected_keywords) if expected_keywords else 0
    return recall, found, missing


def run_eval(results_file=DEFAULT_RESULTS_FILE, enhanced=False):
    mode_label = "enhanced" if enhanced else "default"
    questions = load_test_questions()
    print(f"=== RAG Evaluation ({len(questions)} questions, mode: {mode_label}) ===\n")

    print("Loading vector store...")
    vectorstore = load_vectorstore()
    num_chunks = vectorstore._collection.count()
    chain, retriever = build_chain(vectorstore, enhanced=enhanced)
    print(f"Ready. ({num_chunks} chunks in store)\n")

    results = []
    totals = {"retrieval_hit": 0, "retrieval_recall": 0, "keyword_recall": 0}

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question']}")

        start = time.time()
        answer, docs = query(q["question"], chain, retriever, enhanced=enhanced)
        elapsed = time.time() - start

        hit, ret_recall, ret_pages = eval_retrieval(docs, q["expected_pages"])
        kw_recall, kw_found, kw_missing = eval_answer(answer, q["expected_keywords"])

        result = {
            "question": q["question"],
            "category": q["category"],
            "retrieval_hit": hit,
            "retrieval_recall": round(ret_recall, 2),
            "keyword_recall": round(kw_recall, 2),
            "retrieved_pages": ret_pages,
            "expected_pages": q["expected_pages"],
            "keywords_found": kw_found,
            "keywords_missing": kw_missing,
            "answer": answer,
            "time_seconds": round(elapsed, 1),
        }
        results.append(result)

        totals["retrieval_hit"] += hit
        totals["retrieval_recall"] += ret_recall
        totals["keyword_recall"] += kw_recall

        # short per-question summary
        status = "PASS" if hit and kw_recall >= 0.5 else "FAIL"
        print(f"  {status} | ret_hit={hit} ret_recall={ret_recall:.2f} kw_recall={kw_recall:.2f} | {elapsed:.1f}s")
        if kw_missing:
            print(f"  Missing keywords: {kw_missing}")
        print()

    # overall summary
    n = len(questions)
    summary = {
        "total_questions": n,
        "num_chunks": num_chunks,
        "avg_retrieval_hit": round(totals["retrieval_hit"] / n, 2),
        "avg_retrieval_recall": round(totals["retrieval_recall"] / n, 2),
        "avg_keyword_recall": round(totals["keyword_recall"] / n, 2),
    }

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Questions:          {n}")
    print(f"  Mode:               {mode_label}")
    print(f"  Retrieval Hit Rate: {summary['avg_retrieval_hit']:.0%}")
    print(f"  Retrieval Recall:   {summary['avg_retrieval_recall']:.0%}")
    print(f"  Keyword Recall:     {summary['avg_keyword_recall']:.0%}")
    print()

    # save results
    output = {"summary": summary, "results": results}
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {results_file}")

    return output


if __name__ == "__main__":
    enhanced = "--enhanced" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--enhanced"]
    results_file = args[0] if args else DEFAULT_RESULTS_FILE
    run_eval(results_file, enhanced=enhanced)
