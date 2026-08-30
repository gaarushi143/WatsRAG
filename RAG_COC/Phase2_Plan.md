# Phase 2 Plan — Smarter Retrieval & Evaluation

## Context
The Phase 1 RAG pipeline works but uses a single naive chunking strategy (1000-char fixed chunks) with no way to measure quality. Before making improvements, we need a way to measure accuracy. Then we systematically test different chunking strategies and pick the best one.

## Part 1: Evaluation Framework

### Files to create
- `test_questions.json` — 10 Q&A pairs with expected answers and source pages
- `eval.py` — runs all test questions, measures retrieval + answer quality, prints a scorecard

### test_questions.json format
```json
[
  {
    "question": "What troops are recommended for TH9?",
    "expected_keywords": ["GoWiPe", "QW Hogs", "LavaLoon"],
    "expected_pages": [11],
    "category": "factual"
  }
]
```

Categories: `factual` (specific facts), `strategy` (advice/recommendations), `comparison` (contrasting options), `specific_th` (TH-level-specific)

### Questions to generate (~10, covering the PDF content)
1. What troops unlock at TH7? → page 2 (factual)
2. What strategies are recommended for TH9? → page 11 (specific_th)
3. How does the Freeze Spell work? → pages 38, 43 (factual)
4. What is the Inferno Tower and how does it work? → page 71+ (factual)
5. What are Village Guards? → page 91+ (factual)
6. When should you upgrade your Town Hall? → pages 107-109 (strategy)
7. What is the LavaLoon attack strategy? → page 21+ (strategy)
8. What does the Mortar defense target? → page 6 (factual)
9. How do shields work in Clash of Clans? → page 91 (factual)
10. What is the Hybrid attack strategy? → pages 21+ (strategy)

User will review and adjust these before we lock them in.

### eval.py metrics

**Retrieval metrics (per question):**
- `retrieval_hit`: did ANY of the expected pages appear in the top-k retrieved chunks? (1 or 0)
- `retrieval_recall`: fraction of expected pages found in top-k (e.g., expected [11, 38], found [11] → 0.5)
- `retrieval_precision`: fraction of top-k chunks that came from expected pages

**Answer metrics (per question):**
- `keyword_recall`: fraction of expected keywords found in the LLM answer (case-insensitive)

**Overall scores:**
- Average retrieval hit rate across all questions
- Average retrieval recall
- Average keyword recall
- Per-question breakdown table

### eval.py flow
1. Load test_questions.json
2. Load vectorstore + build chain (reuse from query.py)
3. For each question:
   - Retrieve top-k docs → compute retrieval metrics
   - Run full chain → compute keyword recall
   - Print per-question scores
4. Print summary table with overall averages
5. Save results to `eval_results.json` for comparison across strategies

## Part 2: Chunking Strategy Experiments

### Approach
Modify `ingest.py` to accept a `--strategy` flag. Each strategy changes how chunks are created. The eval flow becomes:
```
python3 ingest.py --strategy <name>   # re-ingests with new strategy
python3 eval.py                        # measures quality
```

### 5 strategies to test

**1. baseline** (current)
- CHUNK_SIZE=1000, CHUNK_OVERLAP=100
- Already ingested, 212 chunks

**2. small_chunks**
- CHUNK_SIZE=500, CHUNK_OVERLAP=50
- More focused embeddings, may improve precision but lose surrounding context

**3. large_chunks**
- CHUNK_SIZE=1500, CHUNK_OVERLAP=200
- More context per chunk, may help strategy questions but dilute retrieval for specific facts

**4. section_aware**
- Parse section headers from the PDF (e.g., "TROOPS", "DEFENSES", "TOWN HALL LEVELS")
- Keep each section as one document, then sub-split large sections with RecursiveCharacterTextSplitter
- Chunks respect topic boundaries instead of cutting mid-topic
- Add `section` to metadata

**5. metadata_enriched**
- Same chunking as baseline, but scan each chunk's text for:
  - Town Hall mentions (regex for "TH\d+" or "Town Hall \d+") → add `town_hall` metadata
  - Troop/spell/defense names → add `topic` metadata tag
- Enables filtered retrieval in query.py (e.g., filter by TH level when the question mentions one)

### Comparison workflow
Create `experiment.py` that automates:
1. For each strategy: clear chroma_db → ingest → eval → save results
2. Print a side-by-side comparison table at the end

## Files to create/modify
| File | Action |
|---|---|
| `test_questions.json` | Create — 10 test Q&A pairs |
| `eval.py` | Create — evaluation script |
| `ingest.py` | Modify — add `--strategy` flag with 5 strategies |
| `query.py` | Modify — support metadata filtering for strategy 5 |
| `experiment.py` | Create — automates running all strategies and comparing |

## Verification
1. Run `python3 eval.py` on current baseline → get baseline scores
2. Run `python3 experiment.py` → runs all 5 strategies, outputs comparison table
3. Review which strategy scores best on retrieval hit rate and keyword recall
4. Lock in the winning strategy as the new default
