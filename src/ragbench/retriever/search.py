from __future__ import annotations

"""Stateless search utilities for the retrieval engine (glass-box design).

Public API
----------
- dense_search(query_vector, faiss_index, top_k) -> (scores, indices)
- keyword_search(query, datastore, top_k) -> (scores, indices)  # placeholder
- combine_results_rrf(results, k=60) -> (scores, indices)

Notes
-----
* `dense_search` expects a **1D** L2-normalized query vector (np.float32). It will
  reshape and dispatch to FAISS. Returned `scores` are the raw FAISS distances:
  - For IndexFlatIP (recommended for cosine via L2-normalized vectors), these are
    **inner products** (a.k.a. cosine similarities).
  - For L2 indexes, these are **squared L2 distances** (lower is better).
* `keyword_search` is a documented placeholder for a future BM25/SPLADE-based
  sparse retrieval implementation; it currently returns empty results.
* `combine_results_rrf` implements Reciprocal Rank Fusion for hybrid search.
"""

from typing import List, Tuple

import numpy as np
import faiss  # type: ignore

# Local import for type hints only; avoids heavy imports at module load elsewhere.
try:  # pragma: no cover - type hint only
    from .datastore import DataStore  # noqa: F401
except Exception:  # pragma: no cover
    DataStore = object  # fallback for type checking in isolation


# --------------------------- helpers ---------------------------

def _prepare_query_vector(query_vector: np.ndarray, expected_dim: int) -> np.ndarray:
    """Validate and reshape a 1D numpy vector for FAISS search.

    Ensures dtype=float32, contiguous layout, and shape=(1, d).

    Raises
    ------
    ValueError
        If the vector is not 1D, has wrong dimensionality, or contains NaNs/inf.
    """
    if not isinstance(query_vector, np.ndarray):
        raise ValueError("query_vector must be a numpy.ndarray")

    if query_vector.ndim != 1:
        raise ValueError(f"query_vector must be 1D; got shape {query_vector.shape}")

    if query_vector.size != expected_dim:
        raise ValueError(
            f"Dimensionality mismatch: index expects d={expected_dim}, got {query_vector.size}"
        )

    if not np.isfinite(query_vector).all():
        raise ValueError("query_vector contains NaN or Inf values")

    # Cast and ensure contiguous float32 (FAISS requirement)
    q = np.asarray(query_vector, dtype=np.float32)
    if not q.flags.c_contiguous:
        q = np.ascontiguousarray(q)

    # Reshape to (1, d) for a single-query search
    return q.reshape(1, expected_dim)


# --------------------------- public API ---------------------------

def dense_search(
    query_vector: np.ndarray,
    faiss_index: faiss.Index,
    top_k: int,
) -> Tuple[List[float], List[int]]:
    """Run dense ANN search against a FAISS index.

    Parameters
    ----------
    query_vector : np.ndarray
        1D L2-normalized vector (float32 preferred). For cosine similarity, use
        IndexFlatIP with L2-normalized vectors.
    faiss_index : faiss.Index
        A pre-built FAISS index (e.g., IndexFlatIP). Must be in RAM.
    top_k : int
        Number of nearest neighbors to return. Will be clipped to index size.

    Returns
    -------
    (scores, indices) : Tuple[List[float], List[int]]
        Scores and integer row indices as returned by FAISS for a single query.
        Note: For IP indexes, higher score = more similar; for L2, lower is better.

    Raises
    ------
    ValueError
        If `top_k` <= 0, the index is empty, or the query vector is invalid.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if getattr(faiss_index, "ntotal", 0) == 0:
        # Empty index; explicit glass-box behavior
        return [], []

    d = int(getattr(faiss_index, "d"))  # expected dimensionality
    q = _prepare_query_vector(query_vector, d)

    # Clip top_k to the index size to avoid FAISS overhead
    k = int(min(top_k, faiss_index.ntotal))

    distances, indices = faiss_index.search(q, k)  # shapes: (1, k)

    # Convert to native Python lists
    scores_list = distances[0].astype(float).tolist()
    indices_list = indices[0].astype(int).tolist()

    return scores_list, indices_list


def keyword_search(
    query: str,
    datastore: "DataStore",
    top_k: int,
) -> Tuple[List[float], List[int]]:
    """Placeholder for sparse keyword retrieval (BM25/SPLADE) used in hybrid search.

    This function intentionally returns empty results for now, serving as a
    well-documented seam where a production BM25 implementation (e.g., bm25s or
    Pyserini/Lucene) will be integrated. The intended behavior is:

      1) Tokenize/normalize the input `query` (lowercasing, basic cleanup).
      2) Retrieve top_k candidate chunk_ids from an inverted index built over
         a chosen text field (e.g., `embedding_text` or `text`).
      3) Map those chunk_ids to row indices using `DataStore` and return scores
         (BM25 scores) alongside integer indices suitable for rank fusion (RRF)
         with dense search results.

    Parameters
    ----------
    query : str
        The raw user query.
    datastore : DataStore
        The DataStore instance, used for ID→row index resolution in the future.
    top_k : int
        The number of keyword-based candidates to fetch.

    Returns
    -------
    (scores, indices) : Tuple[List[float], List[int]]
        Currently empty lists; to be populated by a future sparse retriever.
    """
    _ = (query, datastore, top_k)  # explicit to silence linters
    return [], []


def combine_results_rrf(
    results: List[Tuple[List[float], List[int]]],
    k: int = 60
) -> Tuple[List[float], List[int]]:
    """Combine multiple search result lists using Reciprocal Rank Fusion (RRF).

    RRF is an algorithm that combines ranked lists from different sources
    without needing to tune parameters or normalize scores. It computes a new
    score for each document based on its rank in each list. The formula for a
    document's RRF score is the sum of 1 / (k + rank) for each list it appears in.

    Parameters
    ----------
    results : List[Tuple[List[float], List[int]]]
        A list of search result tuples. Each tuple must be `(scores, indices)`,
        where `indices` are the integer document/chunk indices. The scores are
        not used by RRF, only the rank (position) of each index.
    k : int, optional
        A constant used in the RRF formula to mitigate the impact of high ranks
        from a single system. The original paper suggests k=60. Default is 60.

    Returns
    -------
    Tuple[List[float], List[int]]
        A single tuple of `(combined_scores, combined_indices)` sorted by the
        RRF score in descending order.

    Example
    -------
    >>> dense_indices = [101, 102, 103]
    >>> sparse_indices = [102, 104, 101]
    >>> combined_scores, combined_indices = combine_results_rrf(
    ...     [(None, dense_indices), (None, sparse_indices)]
    ... )
    >>> print(combined_indices)
    [102, 101, 104, 103]
    # RRF scores (approx):
    # doc 102: (1/(60+2)) + (1/(60+1)) = 0.0161 + 0.0164 = 0.0325
    # doc 101: (1/(60+1)) + (1/(60+3)) = 0.0164 + 0.0159 = 0.0323
    # doc 104: (1/(60+2)) = 0.0161
    # doc 103: (1/(60+3)) = 0.0159
    """
    if not results:
        return [], []
    
    # Dictionary to accumulate RRF scores for each document index
    rrf_scores = {}
    
    for scores_list, indices_list in results:
        if not indices_list:  # Skip empty result lists
            continue
            
        # Calculate RRF contribution for each document in this ranked list
        for rank, doc_index in enumerate(indices_list):
            # RRF formula: 1 / (k + rank), where rank is 1-indexed
            rrf_contribution = 1.0 / (k + rank + 1)
            rrf_scores[doc_index] = rrf_scores.get(doc_index, 0.0) + rrf_contribution
    
    if not rrf_scores:
        return [], []
    
    # Sort by RRF score (descending) and extract indices and scores
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    combined_indices = [doc_index for doc_index, _ in sorted_items]
    combined_scores = [rrf_score for _, rrf_score in sorted_items]
    
    return combined_scores, combined_indices
