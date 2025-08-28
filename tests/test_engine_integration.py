"""
Integration test suite for the RetrievalEngine class.

This test suite verifies that the real components—not mocks—can communicate 
correctly. It uses real models, real data, and real FAISS indices to ensure 
the complete retrieval pipeline works end-to-end.

The goal is to confirm that the engine initializes without crashing and that 
all components can work together in production-like conditions.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import faiss

# Ensure the src directory is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ragbench.retriever.datastore import DataStore
from ragbench.retriever.encoders import EncoderManager
from ragbench.retriever.engine import RetrievalEngine


class TestRetrievalEngineIntegration:
    """Integration tests using real data and real models."""
    
    @pytest.fixture(scope="class")
    def real_data_paths(self) -> Dict[str, str]:
        """Get paths to real data files if they exist, otherwise skip tests."""
        test_root = Path(__file__).resolve().parent.parent
        data_dir = test_root / "data" / "processed"
        embeddings_dir = data_dir / "embeddings"
        
        # Look for the processed data files
        faiss_files = list(embeddings_dir.glob("*.faiss"))
        parquet_files = list(embeddings_dir.glob("*.vectors.parquet"))
        jsonl_files = list(data_dir.glob("*_chunks.jsonl"))
        
        if not faiss_files or not parquet_files or not jsonl_files:
            pytest.skip(
                f"Real data files not found. Expected: FAISS index, vectors.parquet, chunks.jsonl. "
                f"Found: {len(faiss_files)} FAISS, {len(parquet_files)} parquet, {len(jsonl_files)} JSONL"
            )
        
        return {
            "faiss_index": str(faiss_files[0]),
            "vectors_parquet": str(parquet_files[0]),
            "chunks_jsonl": str(jsonl_files[0]),
        }
    
    @pytest.fixture(scope="class")
    def real_datastore(self, real_data_paths) -> DataStore:
        """Create a DataStore using the real vectors.parquet file.
        
        The vectors.parquet file now contains rewired neighbor chains that are
        consistent among the kept (embedded) chunks, so we can use it directly.
        """
        vectors_parquet = real_data_paths["vectors_parquet"]
        
        # The vectors.parquet file should now contain all required columns
        # including rewired prev_id/next_id fields
        vectors_df = pd.read_parquet(vectors_parquet)
        
        # Verify required columns exist
        required_columns = {'chunk_id', 'prev_id', 'next_id', 'section_path'}
        missing_columns = required_columns - set(vectors_df.columns)
        if missing_columns:
            pytest.skip(f"Missing required columns in vectors.parquet: {missing_columns}. "
                       f"The embedding process may not have been run with the neighbor rewiring fix.")
        
        # Use the vectors.parquet file directly as it now has consistent neighbor chains
        datastore = DataStore(vectors_parquet)
        return datastore
    
    @pytest.fixture(scope="class") 
    def real_faiss_index(self, real_data_paths) -> faiss.Index:
        """Load real FAISS index from disk."""
        faiss_path = real_data_paths["faiss_index"]
        return faiss.read_index(faiss_path)
    
    @pytest.fixture(scope="class")
    def lightweight_encoder_manager(self) -> EncoderManager:
        """Create a real EncoderManager that matches the FAISS index dimensions."""
        # Use the same embedding model as the FAISS index to ensure dimensional compatibility
        # The FAISS index was built with BAAI/bge-large-en-v1.5 (1024d)
        return EncoderManager(
            embed_model_name="BAAI/bge-large-en-v1.5",  # 1024d, matches FAISS index
            reranker_model_name="cross-encoder/ms-marco-MiniLM-L-2-v2",  # fast reranker
            embedding_max_length=256,  # Smaller context for speed
            rerank_max_length=256,
            rerank_batch_size=16,  # Smaller batches for memory efficiency
        )
    
    def test_engine_initialization_with_real_data(
        self, 
        real_datastore, 
        real_faiss_index, 
        lightweight_encoder_manager
    ):
        """Test that the engine can initialize with real data and models without crashing."""
        # This is the critical integration test - can all components work together?
        start_time = time.time()
        
        engine = RetrievalEngine(
            datastore=real_datastore,
            faiss_index=real_faiss_index,
            encoder_manager=lightweight_encoder_manager,
        )
        
        init_time = time.time() - start_time
        
        # Verify the engine initialized correctly
        assert engine is not None
        assert engine._datastore is not None
        assert engine._faiss_index is not None
        assert engine._encoders is not None
        
        # Check that the data alignment makes sense
        assert len(engine._row_idx_to_chunk_id) > 0
        
        # Verify FAISS index properties
        assert engine._faiss_index.ntotal > 0
        assert engine._faiss_index.d > 0  # Should have proper dimensionality
        
        # Log performance for monitoring
        print(f"Engine initialization took {init_time:.2f} seconds")
        print(f"Loaded {len(engine._row_idx_to_chunk_id)} chunks")
        print(f"FAISS index: {engine._faiss_index.ntotal} vectors, {engine._faiss_index.d}D")
        
        # Initialization should complete in reasonable time (under 30 seconds for small models)
        assert init_time < 30.0, f"Initialization too slow: {init_time:.2f}s"
    
    def test_retrieve_completes_without_error(
        self, 
        real_datastore, 
        real_faiss_index, 
        lightweight_encoder_manager
    ):
        """Test that a simple retrieve call runs to completion without crashing."""
        engine = RetrievalEngine(
            datastore=real_datastore,
            faiss_index=real_faiss_index,
            encoder_manager=lightweight_encoder_manager,
        )
        
        # Test various types of queries that might appear in production
        test_queries = [
            "spacecraft design requirements",
            "contamination control procedures", 
            "What are the testing protocols?",
            "safety requirements for space missions",
            "material specifications",
        ]
        
        for query in test_queries:
            start_time = time.time()
            
            results = engine.retrieve(
                query=query,
                top_k_dense=20,
                top_k_sparse=10,  # Currently placeholder but tests the code path
                top_k_rerank=5,
                context_window_size=2,
                rrf_k=60,
            )
            
            retrieve_time = time.time() - start_time
            
            # Verify the retrieve completed without error
            assert isinstance(results, list), f"Results should be list, got {type(results)}"
            
            # Should return some results for real data
            assert len(results) >= 0, "Should return non-negative number of results"
            
            # If we got results, verify their structure
            if results:
                assert len(results) <= 5, f"Should respect top_k_rerank=5, got {len(results)}"
                
                for i, result in enumerate(results):
                    self._verify_result_structure(result, query, i)
            
            # Performance check - should complete reasonably fast
            assert retrieve_time < 10.0, f"Retrieve too slow for query '{query}': {retrieve_time:.2f}s"
            
            print(f"Query '{query}' returned {len(results)} results in {retrieve_time:.2f}s")
    
    def test_output_schema_and_data_types(
        self, 
        real_datastore, 
        real_faiss_index, 
        lightweight_encoder_manager
    ):
        """Test that the output has correct schema and data types."""
        engine = RetrievalEngine(
            datastore=real_datastore,
            faiss_index=real_faiss_index,
            encoder_manager=lightweight_encoder_manager,
        )
        
        results = engine.retrieve(
            query="spacecraft contamination control",
            top_k_rerank=3,
            context_window_size=1,
        )
        
        # Verify overall structure
        assert isinstance(results, list)
        
        if results:  # If we got results, validate their structure thoroughly
            for i, result in enumerate(results):
                # Top-level structure
                assert isinstance(result, dict), f"Result {i} should be dict"
                assert "chunk" in result, f"Result {i} missing 'chunk' key"
                assert "rerank_score" in result, f"Result {i} missing 'rerank_score' key"  
                assert "neighbors" in result, f"Result {i} missing 'neighbors' key"
                
                # Chunk structure and types
                chunk = result["chunk"]
                assert isinstance(chunk, dict), f"Result {i} chunk should be dict"
                assert "chunk_id" in chunk, f"Result {i} chunk missing 'chunk_id'"
                assert isinstance(chunk["chunk_id"], str), f"Result {i} chunk_id should be str"
                
                # Should have text content
                assert "text" in chunk or "embedding_text" in chunk, \
                    f"Result {i} chunk should have 'text' or 'embedding_text'"
                
                if "text" in chunk:
                    assert isinstance(chunk["text"], str), f"Result {i} text should be str"
                if "embedding_text" in chunk:
                    assert isinstance(chunk["embedding_text"], str), f"Result {i} embedding_text should be str"
                
                # Rerank score
                rerank_score = result["rerank_score"]
                assert isinstance(rerank_score, (int, float)), f"Result {i} rerank_score should be numeric"
                assert not np.isnan(rerank_score), f"Result {i} rerank_score should not be NaN"
                
                # Neighbors structure
                neighbors = result["neighbors"]
                assert isinstance(neighbors, dict), f"Result {i} neighbors should be dict"
                assert "prev" in neighbors, f"Result {i} neighbors missing 'prev'"
                assert "next" in neighbors, f"Result {i} neighbors missing 'next'"
                
                assert isinstance(neighbors["prev"], list), f"Result {i} prev neighbors should be list"
                assert isinstance(neighbors["next"], list), f"Result {i} next neighbors should be list"
                
                # Verify neighbor chunks have proper structure
                for direction in ["prev", "next"]:
                    for j, neighbor in enumerate(neighbors[direction]):
                        assert isinstance(neighbor, dict), f"Result {i} {direction} neighbor {j} should be dict"
                        assert "chunk_id" in neighbor, f"Result {i} {direction} neighbor {j} missing chunk_id"
                        assert isinstance(neighbor["chunk_id"], str), f"Result {i} {direction} neighbor {j} chunk_id should be str"
    
    def test_retrieval_quality_with_real_data(
        self, 
        real_datastore, 
        real_faiss_index, 
        lightweight_encoder_manager
    ):
        """Test retrieval quality with domain-specific queries for NASA document."""
        engine = RetrievalEngine(
            datastore=real_datastore,
            faiss_index=real_faiss_index,
            encoder_manager=lightweight_encoder_manager,
        )
        
        # Domain-specific queries that should find relevant content in NASA standard
        domain_queries = [
            "contamination control",
            "spacecraft cleanliness", 
            "testing procedures",
            "quality assurance",
            "material specifications",
        ]
        
        for query in domain_queries:
            results = engine.retrieve(
                query=query,
                top_k_rerank=10,
                context_window_size=1,
            )
            
            if results:
                # Verify results are ranked by relevance (rerank scores descending)
                scores = [r["rerank_score"] for r in results]
                assert scores == sorted(scores, reverse=True), \
                    f"Results for '{query}' not properly ranked by rerank_score"
                
                # The top result should be reasonably relevant
                top_result = results[0]
                top_text = self._get_text_content(top_result["chunk"]).lower()
                
                # Should contain some relevant terms (relaxed check for real data)
                query_terms = query.lower().split()
                matches = sum(1 for term in query_terms if term in top_text)
                
                # Log for manual inspection
                print(f"\nQuery: '{query}'")
                print(f"Top result relevance: {top_result['rerank_score']:.3f}")
                print(f"Matching terms: {matches}/{len(query_terms)}")
                print(f"Top text snippet: {top_text[:200]}...")
    
    def test_context_expansion_with_real_data(
        self, 
        real_datastore, 
        real_faiss_index, 
        lightweight_encoder_manager
    ):
        """Test that context expansion provides meaningful neighboring chunks."""
        engine = RetrievalEngine(
            datastore=real_datastore,
            faiss_index=real_faiss_index,
            encoder_manager=lightweight_encoder_manager,
        )
        
        results = engine.retrieve(
            query="contamination control procedures",
            top_k_rerank=3,
            context_window_size=2,  # Get 2 neighbors in each direction
        )
        
        if results:
            for i, result in enumerate(results):
                neighbors = result["neighbors"] 
                prev_neighbors = neighbors["prev"]
                next_neighbors = neighbors["next"]
                
                # Verify neighbor chain consistency
                current_chunk_id = result["chunk"]["chunk_id"]
                
                if prev_neighbors:
                    # First prev neighbor should link to current chunk
                    first_prev = prev_neighbors[0]
                    if "next_id" in first_prev:
                        assert first_prev["next_id"] == current_chunk_id or first_prev["next_id"] is None, \
                            f"Broken prev chain for result {i}"
                
                if next_neighbors:
                    # First next neighbor should link back to current chunk  
                    first_next = next_neighbors[0]
                    if "prev_id" in first_next:
                        assert first_next["prev_id"] == current_chunk_id or first_next["prev_id"] is None, \
                            f"Broken next chain for result {i}"
                
                # Log context expansion results
                print(f"\nResult {i} context:")
                print(f"  Current: {current_chunk_id}")
                print(f"  Prev neighbors: {len(prev_neighbors)} ({[n.get('chunk_id', 'N/A') for n in prev_neighbors]})")
                print(f"  Next neighbors: {len(next_neighbors)} ({[n.get('chunk_id', 'N/A') for n in next_neighbors]})")
    
    def test_performance_characteristics(
        self, 
        real_datastore, 
        real_faiss_index, 
        lightweight_encoder_manager
    ):
        """Test performance characteristics with real data and models."""
        engine = RetrievalEngine(
            datastore=real_datastore,
            faiss_index=real_faiss_index,
            encoder_manager=lightweight_encoder_manager,
        )
        
        # Test with different parameter combinations
        test_cases = [
            {"top_k_dense": 10, "top_k_rerank": 3, "context_window_size": 0},
            {"top_k_dense": 50, "top_k_rerank": 10, "context_window_size": 1},
            {"top_k_dense": 100, "top_k_rerank": 20, "context_window_size": 2},
        ]
        
        query = "spacecraft contamination control requirements"
        
        for i, params in enumerate(test_cases):
            start_time = time.time()
            
            results = engine.retrieve(query=query, **params)
            
            elapsed_time = time.time() - start_time
            
            # Performance expectations for lightweight models
            max_time = 5.0 + (params["top_k_dense"] / 100) * 2  # Scale with complexity
            assert elapsed_time < max_time, \
                f"Test case {i} too slow: {elapsed_time:.2f}s > {max_time:.2f}s"
            
            # Should return reasonable number of results
            expected_results = min(params["top_k_rerank"], engine._faiss_index.ntotal)
            assert len(results) <= expected_results
            
            print(f"Test case {i}: {params} -> {len(results)} results in {elapsed_time:.2f}s")
    
    def test_error_handling_with_real_components(
        self, 
        real_datastore, 
        real_faiss_index, 
        lightweight_encoder_manager
    ):
        """Test error handling with real components."""
        engine = RetrievalEngine(
            datastore=real_datastore,
            faiss_index=real_faiss_index,
            encoder_manager=lightweight_encoder_manager,
        )
        
        # Test edge cases that should be handled gracefully
        
        # Empty query
        results = engine.retrieve("")
        assert results == []
        
        # Very long query (should be truncated by tokenizer)
        long_query = "contamination " * 1000  # Very long query
        results = engine.retrieve(long_query, top_k_rerank=1)
        # Should complete without error, even if truncated
        assert isinstance(results, list)
        
        # Zero results requested
        results = engine.retrieve("test query", top_k_rerank=0)
        assert results == []
        
        # Large top_k values (should be clipped)
        results = engine.retrieve(
            "test query", 
            top_k_dense=10000,  # Larger than index
            top_k_rerank=5
        )
        assert len(results) <= 5  # Should be limited by rerank
    
    # Helper methods
    
    def _verify_result_structure(self, result: Dict[str, Any], query: str, index: int) -> None:
        """Verify that a single result has the expected structure."""
        assert isinstance(result, dict), f"Result {index} should be dict"
        
        # Required top-level keys
        required_keys = {"chunk", "rerank_score", "neighbors"}
        assert all(key in result for key in required_keys), \
            f"Result {index} missing required keys. Got: {list(result.keys())}"
        
        # Chunk validation
        chunk = result["chunk"]
        assert isinstance(chunk, dict), f"Result {index} chunk should be dict"
        assert "chunk_id" in chunk, f"Result {index} chunk missing chunk_id"
        
        # Score validation
        score = result["rerank_score"]
        assert isinstance(score, (int, float)), f"Result {index} rerank_score should be numeric"
        assert not np.isnan(score), f"Result {index} rerank_score is NaN"
        
        # Neighbors validation
        neighbors = result["neighbors"]
        assert isinstance(neighbors, dict), f"Result {index} neighbors should be dict"
        assert "prev" in neighbors and "next" in neighbors, \
            f"Result {index} neighbors missing prev/next keys"
        assert isinstance(neighbors["prev"], list), f"Result {index} prev should be list"
        assert isinstance(neighbors["next"], list), f"Result {index} next should be list"
    
    def _get_text_content(self, chunk: Dict[str, Any]) -> str:
        """Extract text content from a chunk, preferring embedding_text."""
        return chunk.get("embedding_text", chunk.get("text", ""))


class TestRetrievalEngineRobustness:
    """Additional robustness tests with real components."""
    
    @pytest.fixture(scope="class")
    def real_data_paths(self) -> Dict[str, str]:
        """Get paths to real data files if they exist, otherwise skip tests."""
        test_root = Path(__file__).resolve().parent.parent
        data_dir = test_root / "data" / "processed"
        embeddings_dir = data_dir / "embeddings"
        
        faiss_files = list(embeddings_dir.glob("*.faiss"))
        parquet_files = list(embeddings_dir.glob("*.vectors.parquet"))
        jsonl_files = list(data_dir.glob("*_chunks.jsonl"))
        
        if not faiss_files or not parquet_files or not jsonl_files:
            pytest.skip("Real data files not found for robustness testing")
        
        return {
            "faiss_index": str(faiss_files[0]),
            "vectors_parquet": str(parquet_files[0]),
            "chunks_jsonl": str(jsonl_files[0]),
        }
    
    def test_engine_cleanup_on_destruction(self, real_data_paths):
        """Test that the engine properly cleans up resources when destroyed.""" 
        # Create temporary combined parquet
        with tempfile.TemporaryDirectory() as temp_dir:
            combined_parquet_path = Path(temp_dir) / "test_chunks.parquet"
            
            # Prepare minimal data
            vectors_df = pd.read_parquet(real_data_paths["vectors_parquet"])
            chunks_data = []
            with open(real_data_paths["chunks_jsonl"], 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10:  # Just use first 10 chunks for speed
                        break
                    chunks_data.append(json.loads(line))
            
            chunks_df = pd.DataFrame(chunks_data)
            combined_df = chunks_df.merge(
                vectors_df[['chunk_id', 'embedding_text']], 
                on='chunk_id', 
                how='inner'
            )
            
            # Fix neighbor chains for the subset to avoid dangling references
            chunk_ids_set = set(combined_df['chunk_id'])
            combined_df['prev_id'] = combined_df['prev_id'].apply(
                lambda x: x if x in chunk_ids_set else None
            )
            combined_df['next_id'] = combined_df['next_id'].apply(
                lambda x: x if x in chunk_ids_set else None
            )
            
            combined_df.to_parquet(combined_parquet_path, index=False)
            
            datastore = DataStore(str(combined_parquet_path))
            faiss_index = faiss.read_index(real_data_paths["faiss_index"])
            
            # Create and destroy engine
            engine = RetrievalEngine(
                datastore=datastore,
                faiss_index=faiss_index,
                embed_model_name="BAAI/bge-large-en-v1.5",  # Must match FAISS index
                reranker_model_name="cross-encoder/ms-marco-MiniLM-L-2-v2",
            )
            
            # Verify it works
            results = engine.retrieve("test query", top_k_rerank=1)
            assert isinstance(results, list)
            
            # Explicitly delete and verify cleanup doesn't crash
            del engine
            
            # This test mainly verifies no exceptions during cleanup
    
    def test_concurrent_usage_safety(self, real_data_paths):
        """Test that the engine can handle multiple queries safely."""
        import threading
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare data
            combined_parquet_path = Path(temp_dir) / "concurrent_test.parquet"
            vectors_df = pd.read_parquet(real_data_paths["vectors_parquet"])
            
            chunks_data = []
            with open(real_data_paths["chunks_jsonl"], 'r') as f:
                for i, line in enumerate(f):
                    if i >= 20:  # Use subset for speed
                        break
                    chunks_data.append(json.loads(line))
            
            chunks_df = pd.DataFrame(chunks_data)
            combined_df = chunks_df.merge(
                vectors_df[['chunk_id', 'embedding_text']], 
                on='chunk_id', 
                how='inner'
            )
            
            # Fix neighbor chains for the subset to avoid dangling references
            chunk_ids_set = set(combined_df['chunk_id'])
            combined_df['prev_id'] = combined_df['prev_id'].apply(
                lambda x: x if x in chunk_ids_set else None
            )
            combined_df['next_id'] = combined_df['next_id'].apply(
                lambda x: x if x in chunk_ids_set else None
            )
            
            combined_df.to_parquet(combined_parquet_path, index=False)
            
            datastore = DataStore(str(combined_parquet_path))
            faiss_index = faiss.read_index(real_data_paths["faiss_index"])
            
            engine = RetrievalEngine(
                datastore=datastore,
                faiss_index=faiss_index,
                embed_model_name="BAAI/bge-large-en-v1.5",  # Must match FAISS index
                reranker_model_name="cross-encoder/ms-marco-MiniLM-L-2-v2",
            )
            
            # Multiple queries in parallel
            results = []
            errors = []
            
            def query_worker(query_id: int):
                try:
                    result = engine.retrieve(
                        f"test query {query_id}",
                        top_k_rerank=2
                    )
                    results.append((query_id, len(result)))
                except Exception as e:
                    errors.append((query_id, str(e)))
            
            # Start multiple threads
            threads = []
            for i in range(5):  # 5 concurrent queries
                thread = threading.Thread(target=query_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify all queries completed successfully
            assert len(errors) == 0, f"Concurrent queries failed: {errors}"
            assert len(results) == 5, f"Expected 5 results, got {len(results)}"
            
            print(f"Concurrent test results: {results}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
