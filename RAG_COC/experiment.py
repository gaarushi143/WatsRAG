"""
Experiment Runner — Compare Chunking Strategies

For each strategy: clears the vector store, re-ingests with that strategy,
runs the evaluation, and saves per-strategy results. At the end, prints a
side-by-side comparison table.

The enhanced_query strategy uses baseline chunking but switches eval to
enhanced mode (query expansion + TOP_K=10 + tuned prompt).

Each strategy runs in a subprocess to avoid ChromaDB connection conflicts.

Usage: python3 experiment.py
"""

import os
import json
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "experiment_results")
PYTHON = os.path.join(PROJECT_DIR, "venv", "bin", "python3")

STRATEGIES = ["baseline", "small_chunks", "large_chunks", "section_aware", "metadata_enriched", "enhanced_query"]


def run_strategy(strategy):
    """Run ingest + eval for one strategy in a subprocess."""
    results_file = os.path.join(RESULTS_DIR, f"{strategy}.json")

    # ingest with this strategy
    print(f"\n--- Ingesting with strategy: {strategy} ---\n")
    ingest_cmd = [PYTHON, os.path.join(PROJECT_DIR, "ingest.py"), "--strategy", strategy]
    subprocess.run(ingest_cmd, check=True)

    # evaluate — use --enhanced flag for the enhanced_query strategy
    print(f"\n--- Evaluating {strategy} ---\n")
    eval_cmd = [PYTHON, os.path.join(PROJECT_DIR, "eval.py")]
    if strategy == "enhanced_query":
        eval_cmd.append("--enhanced")
    eval_cmd.append(results_file)
    subprocess.run(eval_cmd, check=True)

    # read results
    with open(results_file) as f:
        result = json.load(f)
    return result["summary"]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for strategy in STRATEGIES:
        print("\n" + "=" * 60)
        print(f"  STRATEGY: {strategy}")
        print("=" * 60)

        summary = run_strategy(strategy)
        all_results[strategy] = summary

    # print comparison table
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Strategy':<22} {'Chunks':>6} {'Ret Hit':>8} {'Ret Recall':>11} {'KW Recall':>10}")
    print("-" * 70)

    for strategy in STRATEGIES:
        s = all_results[strategy]
        num_chunks = s.get("num_chunks", "?")

        print(
            f"{strategy:<22} {str(num_chunks):>6} "
            f"{s['avg_retrieval_hit']:>7.0%} "
            f"{s['avg_retrieval_recall']:>10.0%} "
            f"{s['avg_keyword_recall']:>9.0%}"
        )

    print("-" * 70)

    # find best strategy
    best = max(STRATEGIES, key=lambda s: (
        all_results[s]["avg_retrieval_hit"],
        all_results[s]["avg_retrieval_recall"],
        all_results[s]["avg_keyword_recall"],
    ))
    print(f"\nBest overall: {best}")

    # save combined results
    combined_path = os.path.join(RESULTS_DIR, "comparison.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
