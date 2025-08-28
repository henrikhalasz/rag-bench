"""Evaluator for retrieval systems."""

import logging
import time
from typing import Any, Dict, List, Optional

from ..retriever.engine import RetrievalEngine
from .metrics import calculate_hit_rate, calculate_mrr, calculate_ndcg

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    tqdm = None

logger = logging.getLogger(__name__)


class RetrieverEvaluator:
    """Evaluates retrieval performance on question-answering datasets."""
    
    def __init__(self, engine: RetrievalEngine):
        """Initialize with a retrieval engine.
        
        Parameters
        ----------
        engine : RetrievalEngine
            The retrieval engine to evaluate.
        """
        self.engine = engine
    
    def run_evaluation(
        self, 
        eval_dataset: List[Dict[str, Any]], 
        top_k: int,
        *,
        top_k_dense: int = 100,
        top_k_sparse: int = 100,
        rrf_k: int = 60,
        context_window_size: int = 1,
        rerank_candidate_cap: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run evaluation on a dataset.
        
        Parameters
        ----------
        eval_dataset : List[Dict[str, Any]]
            List of evaluation examples, each with "question" and "ground_truth_context".
        top_k : int
            Number of top results to retrieve for evaluation.
        top_k_dense : int
            Number of dense ANN candidates to retrieve from FAISS.
        top_k_sparse : int
            Number of keyword/BM25 candidates.
        rrf_k : int
            Constant for Reciprocal Rank Fusion.
        context_window_size : int
            Neighbor window to include on each side.
        rerank_candidate_cap : Optional[int]
            Explicit cap on candidates passed to reranker.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of evaluation results, each containing question, ground truth,
            retrieved IDs, and calculated metrics.
        """
        results = []
        failed_questions = []
        
        # Use tqdm for progress bar if available
        iterator = tqdm(eval_dataset, desc="Running evaluation") if tqdm else eval_dataset
        
        for i, item in enumerate(iterator):
            try:
                question = item["question"]
                ground_truth_ids = set(item["ground_truth_context"])
                
                # Time the retrieval
                start_time = time.time()
                retrieved = self.engine.retrieve(
                    question, 
                    top_k_rerank=top_k,
                    top_k_dense=top_k_dense,
                    top_k_sparse=top_k_sparse,
                    rrf_k=rrf_k,
                    context_window_size=context_window_size,
                    rerank_candidate_cap=rerank_candidate_cap,
                )
                retrieval_time = time.time() - start_time
                
                retrieved_ids = [doc["chunk"]["chunk_id"] for doc in retrieved]
                
                # Calculate metrics
                hit_rate = calculate_hit_rate(retrieved_ids, ground_truth_ids)
                mrr = calculate_mrr(retrieved_ids, ground_truth_ids)
                ndcg = calculate_ndcg(retrieved_ids, ground_truth_ids, k=top_k)
                
                # Store results
                result = {
                    "question": question,
                    "ground_truth_ids": list(ground_truth_ids),
                    "retrieved_ids": retrieved_ids,
                    "hit_rate": hit_rate,
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "latency_seconds": retrieval_time,
                }
                results.append(result)
                
            except Exception as e:
                logger.error(
                    "Failed to evaluate question %d: %s. Question: %s", 
                    i, str(e), item.get("question", "Unknown")
                )
                failed_questions.append({"index": i, "question": item.get("question", "Unknown"), "error": str(e)})
                
                # Add a placeholder result to maintain indexing
                result = {
                    "question": item.get("question", "Unknown"),
                    "ground_truth_ids": list(item.get("ground_truth_context", [])),
                    "retrieved_ids": [],
                    "hit_rate": False,
                    "mrr": 0.0,
                    "ndcg": 0.0,
                    "latency_seconds": 0.0,
                    "error": str(e),
                }
                results.append(result)
        
        if failed_questions:
            logger.warning(
                "Evaluation completed with %d failed questions out of %d total", 
                len(failed_questions), len(eval_dataset)
            )
        
        return results
