from __future__ import annotations

"""Retrieval engine orchestrating hybrid search + rerank + context expansion.

Glass‑box, production‑grade implementation that wires together:
  - DataStore (Parquet‑backed metadata)
  - FAISS dense ANN index (IndexFlatIP recommended with L2‑normalized vectors)
  - EncoderManager (BGE bi‑encoder + cross‑encoder reranker)
  - Stateless search utilities (dense, keyword placeholder, RRF fusion)

Public API
----------
class RetrievalEngine:
    def __init__(...):
        ...
    def retrieve(self, query: str, ...) -> List[dict]:
        ...

Return shape of retrieve()
--------------------------
A list of dicts, each with:
    {
        "chunk": <chunk_row_dict>,
        "rerank_score": float,
        "neighbors": {"prev": [dict, ...], "next": [dict, ...]},
    }

No magic: logging at each stage; explicit parameters; graceful empty results.
"""

from typing import Any, Dict, List, Optional, Optional
import logging
import time

import numpy as np
import faiss  # type: ignore

from .datastore import DataStore
from .encoders import EncoderManager
from .search import dense_search, keyword_search, combine_results_rrf

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Stateful orchestrator for RAG retrieval.

    Parameters
    ----------
    datastore_path : str
        Path to the Parquet metadata file.
    faiss_index_path : str
        Path to the serialized FAISS index.
    embed_model_name : str
        HF id for the embedding (bi‑encoder) model.
    reranker_model_name : str
        HF id for the cross‑encoder reranking model.
    text_field : str
        Chunk dict field that contains text for reranking (default: "embedding_text").
    """

    def __init__(
        self,
        datastore_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        embed_model_name: str = "BAAI/bge-large-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-large",
        text_field: str = "embedding_text",
        # Dependency injection for testing
        datastore: Optional[DataStore] = None,
        encoder_manager: Optional[EncoderManager] = None,
        faiss_index: Optional[faiss.Index] = None,
    ) -> None:
        # -- Load or inject DataStore (memory-mapped Arrow)
        if datastore is not None:
            self._datastore = datastore
        elif datastore_path is not None:
            self._datastore = DataStore(datastore_path)
        else:
            raise ValueError("Either datastore_path or datastore must be provided")

        # -- Load or inject FAISS index (RAM)
        if faiss_index is not None:
            self._faiss_index: faiss.Index = faiss_index
        elif faiss_index_path is not None:
            self._faiss_index: faiss.Index = faiss.read_index(faiss_index_path)
        else:
            raise ValueError("Either faiss_index_path or faiss_index must be provided")
            
        ntotal = int(getattr(self._faiss_index, "ntotal", 0))
        d = int(getattr(self._faiss_index, "d", -1))

        # -- Build an index→chunk_id vector assuming the FAISS index order
        #    aligns with the Parquet row order (standard build procedure for
        #    IndexFlatIP without explicit ID mapping). If your index uses
        #    custom IDs, wrap with IndexIDMap and adapt this mapping logic.
        try:
            self._row_idx_to_chunk_id: List[str] = self._datastore._table[  # type: ignore[attr-defined]
                "chunk_id"
            ].to_pylist()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Unable to build row→chunk_id mapping from DataStore") from exc

        if ntotal != len(self._row_idx_to_chunk_id):
            logger.warning(
                "FAISS ntotal (%d) != number of rows in datastore (%d). Ensure your index is aligned.",
                ntotal,
                len(self._row_idx_to_chunk_id),
            )

        # -- Load or inject models once (device auto‑selected, e.g., MPS on Apple Silicon)
        if encoder_manager is not None:
            self._encoders = encoder_manager
        else:
            self._encoders = EncoderManager(
                embed_model_name=embed_model_name,
                reranker_model_name=reranker_model_name,
                text_field=text_field,
            )

        # -- Log environment info (glass‑box)
        try:
            nthreads = faiss.get_num_threads()
        except Exception:  # pragma: no cover
            nthreads = -1
        logger.info(
            "RetrievalEngine ready | FAISS: d=%s ntotal=%s threads=%s",
            d,
            ntotal,
            nthreads,
        )

    def __del__(self):
        """Clean up resources when the engine is destroyed."""
        try:
            # Explicitly unload models to free GPU memory
            if hasattr(self, '_encoders'):
                del self._encoders
        except Exception:  # pragma: no cover
            # Ignore cleanup errors during destruction
            pass

    # ------------------------- public API -------------------------
    def retrieve(
        self,
        query: str,
        *,
        top_k_dense: int = 100,
        top_k_sparse: int = 100,
        top_k_rerank: int = 25,
        context_window_size: int = 1,
        rrf_k: int = 60,
        rerank_candidate_cap: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full retrieval pipeline for a natural‑language query.

        Steps: encode → dense + sparse → RRF fuse → fetch metadata → rerank →
               context expansion → format & return.

        Parameters
        ----------
        query : str
            The user query.
        top_k_dense : int
            Number of dense ANN candidates to retrieve from FAISS.
        top_k_sparse : int
            Number of keyword/BM25 candidates (currently placeholder hook).
        top_k_rerank : int
            Number of final results to return after cross‑encoder reranking.
        context_window_size : int
            Neighbor window to include on each side (prev/next).
        rrf_k : int
            Constant for Reciprocal Rank Fusion.
        rerank_candidate_cap : Optional[int]
            Explicit cap on how many fused candidates are passed to the reranker.
            If None (default), we compute a predictable cap as
            max(top_k_dense, top_k_sparse, top_k_rerank). This guarantees that
            requesting a larger `top_k_rerank` automatically expands the
            rerank pool (subject to available fused candidates).

        Returns
        -------
        List[Dict[str, Any]]
            Final reranked chunks with neighbors and scores.
        """
        # Input validation
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        if top_k_dense <= 0 or top_k_sparse <= 0:
            raise ValueError("top_k_dense and top_k_sparse must be positive integers")
        
        if top_k_rerank < 0:
            raise ValueError("top_k_rerank must be a non-negative integer")
        
        if context_window_size < 0:
            raise ValueError("context_window_size must be a non-negative integer")
        
        if rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")

        start_time = time.time()
        
        # 1) Encode query to vector (L2‑normalized, float32)
        encode_start = time.time()
        qvec: np.ndarray = self._encoders.encode_query(query)
        logger.debug("Query encoding took %.3fs", time.time() - encode_start)

        # 2) Initial retrieval: dense + sparse (placeholder)
        search_start = time.time()
        dense_scores, dense_indices = dense_search(qvec, self._faiss_index, top_k_dense)
        sparse_scores, sparse_indices = keyword_search(query, self._datastore, top_k_sparse)
        logger.debug("Initial search took %.3fs", time.time() - search_start)

        logger.info(
            "Stage: initial retrieval | dense=%d | sparse=%d",
            len(dense_indices),
            len(sparse_indices),
        )

        # Early exit if nothing retrieved at all
        if not dense_indices and not sparse_indices:
            return []

        # 3) Rank fusion (RRF)
        fusion_start = time.time()
        fused_scores, fused_indices = combine_results_rrf(
            [(dense_scores, dense_indices), (sparse_scores, sparse_indices)], k=rrf_k
        )
        logger.debug("RRF fusion took %.3fs", time.time() - fusion_start)
        logger.info("Stage: fusion | fused_candidates=%d", len(fused_indices))
        if not fused_indices:
            return []

        # 4) Determine candidate pool size for reranker (predictable control)
        if rerank_candidate_cap is not None:
            if rerank_candidate_cap <= 0:
                raise ValueError("rerank_candidate_cap must be a positive integer when provided")
            candidate_cap = rerank_candidate_cap
            cap_reason = "explicit"
        else:
            # Ensure the pool can satisfy top_k_rerank by default
            candidate_cap = max(top_k_dense, top_k_sparse, top_k_rerank)
            cap_reason = "default=max(dense,sparse,rerank)"

        # Clip to available fused candidates to avoid out-of-range
        candidate_cap_effective = min(candidate_cap, len(fused_indices))
        if candidate_cap_effective < candidate_cap:
            logger.warning(
                "Rerank candidate cap reduced from %d to %d due to fused pool size",
                candidate_cap,
                candidate_cap_effective,
            )

        logger.info(
            "Stage: cap | rerank_candidate_cap=%s (%s) | effective=%d",
            candidate_cap,
            cap_reason,
            candidate_cap_effective,
        )

        candidate_indices = fused_indices[:candidate_cap_effective]

        # 5) Map indices → chunk_ids and fetch metadata
        fetch_start = time.time()
        valid_indices = [i for i in candidate_indices if 0 <= i < len(self._row_idx_to_chunk_id)]
        candidate_chunk_ids = [self._row_idx_to_chunk_id[i] for i in valid_indices]

        if not candidate_chunk_ids:
            return []

        candidates = self._datastore.fetch_chunks_by_ids(candidate_chunk_ids)
        logger.debug("Metadata fetch took %.3fs", time.time() - fetch_start)
        logger.info("Stage: fetch | candidates_for_rerank=%d", len(candidates))
        if not candidates:
            return []

        # 6) Cross‑encoder rerank (precision stage)
        rerank_start = time.time()
        reranked = self._encoders.rerank(query, candidates)
        logger.debug("Cross-encoder reranking took %.3fs", time.time() - rerank_start)
        logger.info("Stage: rerank | reranked=%d", len(reranked))
        if not reranked:
            return []

        final = reranked[: max(0, top_k_rerank)]

        # 7) Context expansion for each final hit
        expansion_start = time.time()
        results: List[Dict[str, Any]] = []
        for item in final:
            cid = item.get("chunk_id")
            score = float(item.get("rerank_score", 0.0))
            try:
                neighbors = self._datastore.get_neighbors(cid, window_size=context_window_size) if cid else {"prev": [], "next": []}
            except Exception as exc:
                logger.warning("neighbors lookup failed for %s: %s", cid, exc)
                neighbors = {"prev": [], "next": []}

            results.append({
                "chunk": item,
                "rerank_score": score,
                "neighbors": neighbors,
            })

        logger.debug("Context expansion took %.3fs", time.time() - expansion_start)
        logger.info("Stage: done | returned=%d | total_time=%.3fs", len(results), time.time() - start_time)
        return results
