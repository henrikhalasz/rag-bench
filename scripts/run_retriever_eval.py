#!/usr/bin/env python3
"""Main entry point for running retrieval evaluations."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragbench.retriever.engine import RetrievalEngine
from ragbench.evaluation.evaluator import RetrieverEvaluator


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate aggregate metrics from per-question results."""
    if not results:
        return {"hit_rate": 0.0, "mean_mrr": 0.0, "mean_ndcg": 0.0, "mean_latency": 0.0}
    
    # Filter out failed results (those with errors)
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        logger.warning("No valid results found - all questions failed")
        return {"hit_rate": 0.0, "mean_mrr": 0.0, "mean_ndcg": 0.0, "mean_latency": 0.0}
    
    hit_rate = sum(1 for r in valid_results if r["hit_rate"]) / len(valid_results)
    mean_mrr = sum(r["mrr"] for r in valid_results) / len(valid_results)
    mean_ndcg = sum(r["ndcg"] for r in valid_results) / len(valid_results)
    mean_latency = sum(r["latency_seconds"] for r in valid_results) / len(valid_results)
    
    return {
        "hit_rate": hit_rate,
        "mean_mrr": mean_mrr,
        "mean_ndcg": mean_ndcg,
        "mean_latency": mean_latency,
        "total_questions": len(results),
        "valid_questions": len(valid_results),
        "failed_questions": len(results) - len(valid_results),
    }


def print_summary_table(metrics: Dict[str, float]):
    """Print a formatted summary table of aggregate metrics."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Questions:    {metrics.get('total_questions', 0)}")
    print(f"Valid Results:      {metrics.get('valid_questions', 0)}")
    print(f"Failed Questions:   {metrics.get('failed_questions', 0)}")
    print("-"*50)
    print(f"Hit Rate:           {metrics['hit_rate']:.3f}")
    print(f"Mean MRR:           {metrics['mean_mrr']:.3f}")
    print(f"Mean nDCG:          {metrics['mean_ndcg']:.3f}")
    print(f"Mean Latency:       {metrics['mean_latency']:.3f}s")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation on a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--eval-dataset",
        required=True,
        help="Path to the evaluation dataset (JSONL file)"
    )
    parser.add_argument(
        "--datastore-path",
        required=True,
        help="Path to the Parquet datastore file"
    )
    parser.add_argument(
        "--faiss-index",
        required=True,
        help="Path to the FAISS index file"
    )
    parser.add_argument(
        "--output-path",
        help="Path to save detailed per-question results (JSON file)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to retrieve for evaluation"
    )
    parser.add_argument(
        "--top-k-dense",
        type=int,
        default=100,
        help="Number of dense candidates to retrieve"
    )
    parser.add_argument(
        "--top-k-sparse",
        type=int,
        default=100,
        help="Number of sparse candidates to retrieve"
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant for rank fusion"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate input files exist
        for path_arg, path_value in [
            ("eval-dataset", args.eval_dataset),
            ("datastore-path", args.datastore_path),
            ("faiss-index", args.faiss_index)
        ]:
            if not Path(path_value).exists():
                logger.error("File not found: %s (%s)", path_value, path_arg)
                sys.exit(1)
        
        # Initialize RetrievalEngine
        logger.info("Initializing RetrievalEngine...")
        engine = RetrievalEngine(
            datastore_path=args.datastore_path,
            faiss_index_path=args.faiss_index
        )
        logger.info("Engine initialized successfully")
        
        # Load evaluation dataset
        logger.info("Loading evaluation dataset from %s", args.eval_dataset)
        eval_dataset = load_jsonl(args.eval_dataset)
        logger.info("Loaded %d questions for evaluation", len(eval_dataset))
        
        # Initialize evaluator
        evaluator = RetrieverEvaluator(engine)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.run_evaluation(
            eval_dataset=eval_dataset,
            top_k=args.top_k,
            top_k_dense=args.top_k_dense,
            top_k_sparse=args.top_k_sparse,
            rrf_k=args.rrf_k,
        )
        logger.info("Evaluation completed")
        
        # Calculate aggregate metrics
        aggregate_metrics = calculate_aggregate_metrics(results)
        
        # Print summary
        print_summary_table(aggregate_metrics)
        
        # Save detailed results if output path provided
        if args.output_path:
            logger.info("Saving detailed results to %s", args.output_path)
            output_data = {
                "aggregate_metrics": aggregate_metrics,
                "per_question_results": results,
                "evaluation_config": {
                    "eval_dataset": args.eval_dataset,
                    "datastore_path": args.datastore_path,
                    "faiss_index": args.faiss_index,
                    "top_k": args.top_k,
                    "top_k_dense": args.top_k_dense,
                    "top_k_sparse": args.top_k_sparse,
                    "rrf_k": args.rrf_k,
                }
            }
            
            # Ensure output directory exists
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info("Results saved successfully")
    
    except Exception as e:
        logger.error("Evaluation failed: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
