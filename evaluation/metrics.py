"""RAG Evaluation Metrics: Retrieval, Generation, Citation, and Abstention.

Provides mathematically rigorous, reproducible metrics:
- Retrieval: Recall@K, Precision@K, Mean Reciprocal Rank (MRR), nDCG@K
- Citations: Citation Precision, Citation Coverage
- Operational: Abstention Accuracy (for unanswerable queries)
- Generation: Lexical alignment heuristics and semantic evaluation hooks
"""

import math
import re
from typing import Any, Dict, List, Optional, Set


STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "of", "is", "are", "was", "were", "it", "this", "that", "by", "as",
    "be", "from", "what", "how", "why", "who", "which", "does", "do",
    "then", "so", "than", "into", "over", "after", "before", "about",
}


def tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric word tokens, ignoring stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


# -------------------------------------------------------------
def is_chunk_relevant(chunk_text: str, target: str, token_threshold: float = 0.45) -> bool:
    """
    Determine if a chunk matches a relevant target using substring match or token overlap.
    Retained for legacy heuristic evaluation and text target matching.
    """
    if target.lower() in chunk_text.lower():
        return True

    target_tokens = tokenize(target)
    if not target_tokens:
        return False

    chunk_tokens = tokenize(chunk_text)
    overlap = target_tokens.intersection(chunk_tokens)
    overlap_ratio = len(overlap) / len(target_tokens)
    return overlap_ratio >= token_threshold


# -------------------------------------------------------------
# RETRIEVAL METRICS (GROUNDED IN STABLE CHUNK IDs & TEXT TARGETS)
# -------------------------------------------------------------
def calculate_recall_at_k(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int = 4,
) -> float:
    """
    Calculate Recall@K:
    Proportion of ground-truth relevant chunks present in the top-K retrieved chunks.
    Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|

    Supports both stable chunk IDs and text-target matching.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer greater than 0.")
    if not relevant_chunk_ids or not retrieved_chunk_ids:
        return 0.0

    top_chunks = retrieved_chunk_ids[:k]
    relevant_set = set(relevant_chunk_ids)

    # 1. Check exact chunk ID / item matches first
    matched_targets = 0
    for target in relevant_chunk_ids:
        if target in top_chunks or any(is_chunk_relevant(c, target) for c in top_chunks):
            matched_targets += 1

    return round(matched_targets / len(relevant_chunk_ids), 4)


def calculate_precision_at_k(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int = 4,
) -> float:
    """
    Calculate Precision@K:
    Proportion of top-K retrieved chunks that are relevant.
    Precision@K = |Retrieved_K ∩ Relevant| / K

    Edge cases:
    - If k <= 0: raises ValueError.
    - If retrieved_chunk_ids or relevant_chunk_ids is empty: returns 0.0.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer greater than 0.")
    if not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    top_k_ids = retrieved_chunk_ids[:k]
    hits = sum(1 for cid in top_k_ids if cid in relevant_set)
    return round(hits / k, 4)


def calculate_mrr(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR):
    Reciprocal rank (1 / rank) of the first relevant chunk in the retrieved list.
    Returns 0.0 if no relevant chunk is retrieved or if inputs are empty.
    """
    if not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in relevant_set or any(is_chunk_relevant(cid, target) for target in relevant_chunk_ids):
            return round(1.0 / rank, 4)

    return 0.0


def calculate_ndcg_at_k(
    retrieved_chunk_ids: List[str],
    relevant_chunk_ids: List[str],
    k: int = 4,
    graded_relevance: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (nDCG@K).
    Supports binary relevance (default 1.0 / 0.0) or graded relevance judgments.

    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    nDCG@K = DCG@K / IDCG@K
    """
    if k <= 0:
        raise ValueError("k must be a positive integer greater than 0.")
    if not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    graded = graded_relevance or {}

    # Calculate DCG@K
    dcg = 0.0
    for idx, cid in enumerate(retrieved_chunk_ids[:k], start=1):
        rel = graded.get(cid, 1.0 if cid in relevant_set else 0.0)
        if rel > 0.0:
            dcg += rel / math.log2(idx + 1)

    # Calculate Ideal DCG@K (IDCG)
    ideal_scores = []
    for cid in relevant_set:
        ideal_scores.append(graded.get(cid, 1.0))
    ideal_scores.sort(reverse=True)

    idcg = 0.0
    for idx, rel in enumerate(ideal_scores[:k], start=1):
        idcg += rel / math.log2(idx + 1)

    if idcg == 0.0:
        return 0.0

    return round(dcg / idcg, 4)


# -------------------------------------------------------------
# CITATION METRICS
# -------------------------------------------------------------
def calculate_citation_precision(
    citations: List[Dict[str, Any]],
    relevant_chunk_ids: List[str],
) -> float:
    """
    Calculate Citation Precision:
    Proportion of emitted citations that reference actual ground-truth relevant chunks.
    Citation Precision = |Valid Citations| / |Total Citations|
    """
    if not citations:
        # If query has no relevant chunks and 0 citations were emitted, precision is 1.0 (no false citation)
        return 1.0 if not relevant_chunk_ids else 0.0

    if not relevant_chunk_ids:
        # Emitted citations when none are relevant -> precision 0.0
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    valid_citations = sum(1 for c in citations if c.get("chunk_id") in relevant_set)
    return round(valid_citations / len(citations), 4)


def calculate_citation_coverage(
    citations: List[Dict[str, Any]],
    relevant_chunk_ids: List[str],
) -> float:
    """
    Calculate Citation Coverage (Recall):
    Proportion of ground-truth relevant chunks that were cited in the answer.
    Citation Coverage = |Cited Relevant Chunks| / |Total Relevant Chunks|
    """
    if not relevant_chunk_ids:
        return 1.0 if not citations else 0.0
    if not citations:
        return 0.0

    cited_ids = {c.get("chunk_id") for c in citations if c.get("chunk_id")}
    covered = len(cited_ids.intersection(set(relevant_chunk_ids)))
    return round(covered / len(relevant_chunk_ids), 4)


# -------------------------------------------------------------
# ABSTENTION QUALITY (FOR UNANSWERABLE QUESTIONS)
# -------------------------------------------------------------
REFUSAL_PATTERNS = [
    r"could not find relevant information",
    r"not found in the uploaded sources",
    r"not mentioned in the provided",
    r"cannot answer",
    r"no information available",
    r"no documents have been ingested",
    r"context does not contain",
]


def is_refusal_answer(answer: str) -> bool:
    """Check if generated answer contains a standardized refusal or abstention statement."""
    clean = answer.strip().lower()
    return any(re.search(pat, clean) for pat in REFUSAL_PATTERNS)


def calculate_abstention_accuracy(is_unanswerable: bool, answer: str) -> float:
    """
    Calculate Abstention Accuracy:
    - 1.0 if an unanswerable query was properly refused, or if an answerable query was answered.
    - 0.0 if the model hallucinated an answer to an unanswerable query, or incorrectly refused an answerable one.
    """
    refused = is_refusal_answer(answer)
    if is_unanswerable:
        return 1.0 if refused else 0.0
    else:
        return 0.0 if refused else 1.0


# -------------------------------------------------------------
# HEURISTIC LEXICAL METRICS (UNBIASED, NO ARBITRARY MULTIPLIERS)
# -------------------------------------------------------------
def calculate_heuristic_faithfulness(
    generated_answer: str,
    retrieved_contexts: List[str],
    support_threshold: float = 0.5,
) -> float:
    """
    Heuristic lexical alignment metric measuring sentence-level word overlap with retrieved context.
    NOTE: Lexical overlap is an approximate heuristic and is NOT proof of semantic factual truth.
    """
    if not generated_answer or not retrieved_contexts:
        return 0.0

    all_context = " ".join(retrieved_contexts)
    context_tokens = tokenize(all_context)
    if not context_tokens:
        return 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]\s+", generated_answer) if s.strip()]
    if not sentences:
        return 0.0

    supported_count = 0
    for sentence in sentences:
        sent_tokens = tokenize(sentence)
        if not sent_tokens:
            continue
        overlap = sent_tokens.intersection(context_tokens)
        support_ratio = len(overlap) / len(sent_tokens)
        if support_ratio >= support_threshold:
            supported_count += 1

    return round(supported_count / len(sentences), 4)


def calculate_heuristic_answer_relevancy(
    question: str,
    generated_answer: str,
) -> float:
    """
    Calculate Lexical Answer Relevancy:
    Evaluates question term coverage and Jaccard similarity.
    Unbiased: does NOT apply arbitrary multipliers. Strictly bounded in [0.0, 1.0].
    """
    q_tokens = tokenize(question)
    a_tokens = tokenize(generated_answer)

    if not q_tokens or not a_tokens:
        return 0.0

    overlap = q_tokens.intersection(a_tokens)
    coverage = len(overlap) / len(q_tokens)
    jaccard = len(overlap) / len(q_tokens.union(a_tokens))

    # Balanced harmonic/weighted formulation bounded between 0.0 and 1.0
    relevancy = (0.7 * coverage) + (0.3 * jaccard)
    return round(min(1.0, max(0.0, relevancy)), 4)


def calculate_heuristic_context_relevancy(
    question: str,
    retrieved_contexts: List[str],
    min_token_overlap: int = 2,
) -> float:
    """
    Calculate Heuristic Context Relevancy:
    Measures proportion of retrieved chunks containing at least min_token_overlap
    salient query tokens (preventing single-stopword false positives).
    """
    if not retrieved_contexts:
        return 0.0

    q_tokens = tokenize(question)
    if not q_tokens:
        return 0.0

    relevant_chunks = 0
    for chunk in retrieved_contexts:
        c_tokens = tokenize(chunk)
        overlap = q_tokens.intersection(c_tokens)
        if len(overlap) >= min(min_token_overlap, len(q_tokens)):
            relevant_chunks += 1

    return round(relevant_chunks / len(retrieved_contexts), 4)


# Backward-compatibility aliases with warnings/docstrings
calculate_faithfulness = calculate_heuristic_faithfulness
calculate_answer_relevancy = calculate_heuristic_answer_relevancy
calculate_context_relevancy = calculate_heuristic_context_relevancy
