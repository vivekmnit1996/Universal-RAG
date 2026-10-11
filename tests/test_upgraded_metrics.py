"""Comprehensive Tests for Upgraded Evaluation Metrics, Citations, and Abstention."""

import json
from pathlib import Path
import pytest

from evaluation.judge import LLMJudge
from evaluation.metrics import (
    calculate_abstention_accuracy,
    calculate_citation_coverage,
    calculate_citation_precision,
    calculate_heuristic_answer_relevancy,
    calculate_heuristic_context_relevancy,
    calculate_heuristic_faithfulness,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    is_refusal_answer,
    tokenize,
)


# -------------------------------------------------------------
# RETRIEVAL METRICS & EDGE CASES
# -------------------------------------------------------------
def test_recall_at_k_exact_and_edge_cases():
    retrieved = ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    relevant = ["chunk-2", "chunk-4", "chunk-99"]

    # At k=1: "chunk-1" -> 0/3 = 0.0
    assert calculate_recall_at_k(retrieved, relevant, k=1) == 0.0

    # At k=2: "chunk-1", "chunk-2" -> 1/3 = 0.3333
    assert calculate_recall_at_k(retrieved, relevant, k=2) == 0.3333

    # At k=4: "chunk-1", "chunk-2", "chunk-3", "chunk-4" -> 2/3 = 0.6667
    assert calculate_recall_at_k(retrieved, relevant, k=4) == 0.6667

    # Edge case: empty retrieved
    assert calculate_recall_at_k([], relevant, k=4) == 0.0

    # Edge case: empty relevant (unanswerable query)
    assert calculate_recall_at_k(retrieved, [], k=4) == 0.0

    # Edge case: duplicate retrieved chunk IDs
    dup_retrieved = ["chunk-2", "chunk-2", "chunk-4"]
    assert calculate_recall_at_k(dup_retrieved, relevant, k=4) == 0.6667

    # Invalid K
    with pytest.raises(ValueError):
        calculate_recall_at_k(retrieved, relevant, k=0)
    with pytest.raises(ValueError):
        calculate_recall_at_k(retrieved, relevant, k=-1)


def test_precision_at_k_exact_and_edge_cases():
    retrieved = ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    relevant = ["chunk-2", "chunk-4"]

    # At k=1: 0/1 = 0.0
    assert calculate_precision_at_k(retrieved, relevant, k=1) == 0.0

    # At k=2: 1/2 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=2) == 0.5

    # At k=4: 2/4 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=4) == 0.5

    # Edge cases
    assert calculate_precision_at_k([], relevant, k=4) == 0.0
    assert calculate_precision_at_k(retrieved, [], k=4) == 0.0

    with pytest.raises(ValueError):
        calculate_precision_at_k(retrieved, relevant, k=0)


def test_mrr_exact_and_edge_cases():
    # Rank 1 hit -> MRR = 1.0
    assert calculate_mrr(["doc-A", "doc-B"], ["doc-A"]) == 1.0

    # Rank 3 hit -> MRR = 1/3 = 0.3333
    assert calculate_mrr(["doc-X", "doc-Y", "doc-Z"], ["doc-Z"]) == 0.3333

    # No hit -> MRR = 0.0
    assert calculate_mrr(["doc-1", "doc-2"], ["doc-99"]) == 0.0

    # Edge cases
    assert calculate_mrr([], ["doc-1"]) == 0.0
    assert calculate_mrr(["doc-1"], []) == 0.0


def test_ndcg_at_k_exact_and_graded():
    retrieved = ["doc-1", "doc-2", "doc-3"]
    relevant = ["doc-1", "doc-2"]

    # Perfect ranking at top positions
    assert calculate_ndcg_at_k(retrieved, relevant, k=2) == 1.0

    # Graded relevance
    graded = {"doc-1": 1.0, "doc-2": 3.0}  # doc-2 is more relevant than doc-1
    # Ideal order is [doc-2 (3.0), doc-1 (1.0)]
    # Retrieved order is [doc-1 (1.0), doc-2 (3.0)]
    ndcg_suboptimal = calculate_ndcg_at_k(retrieved, relevant, k=2, graded_relevance=graded)
    assert 0.0 < ndcg_suboptimal < 1.0

    # Edge cases
    assert calculate_ndcg_at_k([], relevant, k=3) == 0.0
    assert calculate_ndcg_at_k(retrieved, [], k=3) == 0.0
    with pytest.raises(ValueError):
        calculate_ndcg_at_k(retrieved, relevant, k=0)


# -------------------------------------------------------------
# CITATION QUALITY METRICS
# -------------------------------------------------------------
def test_citation_precision_and_coverage():
    citations = [
        {"chunk_id": "chunk-10", "label": "Doc A - Page 1"},
        {"chunk_id": "chunk-20", "label": "Doc B - Page 2"},
    ]
    relevant = ["chunk-10", "chunk-30"]

    # 1 of 2 citations is relevant -> Precision = 0.5
    assert calculate_citation_precision(citations, relevant) == 0.5

    # 1 of 2 relevant targets cited -> Coverage = 0.5
    assert calculate_citation_coverage(citations, relevant) == 0.5

    # Edge case: no citations emitted
    assert calculate_citation_precision([], relevant) == 0.0
    assert calculate_citation_coverage([], relevant) == 0.0

    # Edge case: unanswerable query with 0 relevant chunks and 0 citations -> 1.0 (correct silence)
    assert calculate_citation_precision([], []) == 1.0
    assert calculate_citation_coverage([], []) == 1.0


# -------------------------------------------------------------
# ABSTENTION QUALITY
# -------------------------------------------------------------
def test_abstention_accuracy():
    refusal_answer = "I could not find relevant information in the uploaded sources to answer your question."
    hallucinated_answer = "The stock price of Apple is $250 per share."
    grounded_answer = "Universal RAG uses FAISS and SQLite for persistence."

    assert is_refusal_answer(refusal_answer)
    assert not is_refusal_answer(hallucinated_answer)

    # 1. Unanswerable question properly refused -> 1.0
    assert calculate_abstention_accuracy(is_unanswerable=True, answer=refusal_answer) == 1.0

    # 2. Unanswerable question hallucinated -> 0.0
    assert calculate_abstention_accuracy(is_unanswerable=True, answer=hallucinated_answer) == 0.0

    # 3. Answerable question answered -> 1.0
    assert calculate_abstention_accuracy(is_unanswerable=False, answer=grounded_answer) == 1.0

    # 4. Answerable question falsely refused -> 0.0
    assert calculate_abstention_accuracy(is_unanswerable=False, answer=refusal_answer) == 0.0


# -------------------------------------------------------------
# HEURISTIC UNBIASED METRICS (NO MULTIPLIERS)
# -------------------------------------------------------------
def test_heuristic_relevancy_strictly_bounded():
    q = "What is FAISS vector similarity search?"
    perfect_ans = "FAISS is a vector similarity search library."
    irrelevant_ans = "Baking sourdough bread requires flour and yeast."

    score_good = calculate_heuristic_answer_relevancy(q, perfect_ans)
    score_bad = calculate_heuristic_answer_relevancy(q, irrelevant_ans)

    assert 0.0 <= score_good <= 1.0
    assert 0.0 <= score_bad <= 1.0
    assert score_good > score_bad


def test_heuristic_context_relevancy_requires_multi_token_overlap():
    q = "What is FAISS vector similarity search?"
    chunk_multi_match = "FAISS provides vector similarity search capabilities."
    chunk_single_stopword = "A system that handles data processing."

    contexts = [chunk_multi_match, chunk_single_stopword]
    score = calculate_heuristic_context_relevancy(q, contexts, min_token_overlap=2)
    # Only 1 chunk has >= 2 query tokens
    assert score == 0.5


# -------------------------------------------------------------
# BENCHMARK DATASET INTEGRITY
# -------------------------------------------------------------
def test_benchmark_dataset_integrity():
    bench_file = Path("evaluation") / "benchmark_dataset.json"
    assert bench_file.exists()

    with open(bench_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    assert len(samples) >= 100

    categories = set()
    for s in samples:
        assert "id" in s
        assert "question" in s
        assert "category" in s
        assert "is_answerable" in s
        assert "relevant_chunk_ids" in s
        assert isinstance(s["is_answerable"], bool)
        categories.add(s["category"])

        if not s["is_answerable"]:
            assert len(s["relevant_chunk_ids"]) == 0

    assert "direct_factual" in categories
    assert "paraphrased" in categories
    assert "multi_chunk" in categories
    assert "unanswerable" in categories
