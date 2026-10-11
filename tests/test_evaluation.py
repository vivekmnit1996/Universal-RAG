"""Tests for evaluation metrics and benchmark runner."""

from evaluation.evaluator import RAGEvaluator
from evaluation.metrics import (
    calculate_answer_relevancy,
    calculate_faithfulness,
    calculate_mrr,
    calculate_recall_at_k,
    is_chunk_relevant,
)


def test_metric_is_chunk_relevant():
    chunk = "Vector databases provide efficient similarity search over dense embeddings."
    assert is_chunk_relevant(chunk, "similarity search over dense embeddings")
    assert not is_chunk_relevant(chunk, "quantum key distribution protocols")


def test_metric_recall_at_k():
    retrieved = [
        "First chunk discussing transformers.",
        "Second chunk discussing attention mechanisms.",
        "Third chunk discussing gradient descent.",
    ]
    targets = ["attention mechanisms", "transformers"]

    # At k=1, only transformers is found -> 1/2 = 0.5
    recall_1 = calculate_recall_at_k(retrieved, targets, k=1)
    assert recall_1 == 0.5

    # At k=2, both are found -> 2/2 = 1.0
    recall_2 = calculate_recall_at_k(retrieved, targets, k=2)
    assert recall_2 == 1.0


def test_metric_mrr():
    retrieved = [
        "Irrelevant chunk on data cleaning.",
        "First relevant chunk on LangGraph workflows.",
        "Second relevant chunk on LangGraph state.",
    ]
    targets = ["LangGraph workflows"]
    # First relevant chunk is at rank 2 -> MRR = 1/2 = 0.5
    mrr = calculate_mrr(retrieved, targets)
    assert mrr == 0.5


def test_metric_faithfulness():
    context = ["LangChain provides document loaders and text splitters for data ingestion."]
    supported_ans = "LangChain provides document loaders for ingestion."
    unsupported_ans = "LangChain was created by Alan Turing in 1950 to solve chess."

    f_good = calculate_faithfulness(supported_ans, context)
    assert f_good == 1.0

    f_bad = calculate_faithfulness(unsupported_ans, context)
    assert f_bad == 0.0


def test_metric_answer_relevancy():
    q = "What is Retrieval-Augmented Generation?"
    relevant_ans = "Retrieval-Augmented Generation is a method that enhances LLM generation with retrieved data."
    irrelevant_ans = "Today's weather is sunny with mild afternoon clouds."

    rel_good = calculate_answer_relevancy(q, relevant_ans)
    rel_bad = calculate_answer_relevancy(q, irrelevant_ans)
    assert rel_good > rel_bad


def test_rag_evaluator_benchmark():
    evaluator = RAGEvaluator()
    results = evaluator.run_evaluation(k=4)

    assert "summary" in results
    assert "per_sample_results" in results
    summary = results["summary"]
    assert summary["total_samples"] == 5
    assert summary["mean_recall@4"] > 0.0
    assert summary["mean_mrr"] > 0.0
