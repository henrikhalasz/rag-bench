"""Evaluation metrics for information retrieval tasks."""

import math


def calculate_hit_rate(retrieved_ids: list, ground_truth_ids: set) -> bool:
    """Return True if any retrieved item is in ground truth."""
    return any(doc_id in ground_truth_ids for doc_id in retrieved_ids)


def calculate_mrr(retrieved_ids: list, ground_truth_ids: set) -> float:
    """Calculate Mean Reciprocal Rank (1/rank of first relevant document)."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in ground_truth_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_ndcg(retrieved_ids: list, ground_truth_ids: set, k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k."""
    if not ground_truth_ids:
        return 0.0
    
    actual_k = min(k, len(retrieved_ids))
    
    # DCG: sum of relevance / log2(position + 1)
    dcg = 0.0
    for i in range(actual_k):
        if retrieved_ids[i] in ground_truth_ids:
            dcg += 1.0 / math.log2(i + 2)
    
    # IDCG: ideal ranking with all relevant docs at top
    num_relevant = min(len(ground_truth_ids), actual_k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))
    
    return dcg / idcg if idcg > 0 else 0.0
