"""
Comprehensive test suite for search module stateless functions.

This test suite validates all search functionality including dense search,
keyword search (placeholder), and Reciprocal Rank Fusion (RRF) combination.
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch

import faiss
import numpy as np
import pytest

# Ensure the src directory is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ragbench.retriever import search


@pytest.fixture
def sample_vectors():
    """Create sample normalized vectors for testing."""
    np.random.seed(42)  # For reproducible tests
    
    # Create 5 sample vectors with dimension 128
    vectors = np.random.randn(5, 128).astype(np.float32)
    
    # L2 normalize them (required for IndexFlatIP cosine similarity)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    
    return vectors


@pytest.fixture
def sample_faiss_index(sample_vectors):
    """Create a FAISS IndexFlatIP with sample vectors."""
    d = sample_vectors.shape[1]  # dimension
    index = faiss.IndexFlatIP(d)  # Inner Product index for cosine similarity
    index.add(sample_vectors)
    return index


@pytest.fixture
def sample_query_vector(sample_vectors):
    """Create a normalized query vector similar to the first sample vector."""
    # Create a query that's similar to the first vector but not identical
    query = sample_vectors[0] * 0.8 + np.random.randn(128).astype(np.float32) * 0.2
    
    # L2 normalize
    query = query / np.linalg.norm(query)
    
    return query


class TestPrepareQueryVector:
    """Test suite for the internal _prepare_query_vector helper function."""
    
    def test_valid_1d_vector(self):
        """Test with valid 1D numpy array."""
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected_dim = 3
        
        result = search._prepare_query_vector(vector, expected_dim)
        
        assert result.shape == (1, 3)
        assert result.dtype == np.float32
        assert result.flags.c_contiguous
        np.testing.assert_array_equal(result[0], vector)
    
    def test_auto_convert_to_float32(self):
        """Test automatic conversion to float32."""
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # float64 input
        expected_dim = 3
        
        result = search._prepare_query_vector(vector, expected_dim)
        
        assert result.dtype == np.float32
        assert result.shape == (1, 3)
    
    def test_ensure_contiguous(self):
        """Test that non-contiguous arrays are made contiguous."""
        vector = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)[0, ::2]  # Non-contiguous
        expected_dim = 2
        
        assert not vector.flags.c_contiguous  # Verify it's non-contiguous
        
        result = search._prepare_query_vector(vector, expected_dim)
        
        assert result.flags.c_contiguous
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result[0], [1.0, 3.0])
    
    def test_invalid_input_type(self):
        """Test error with non-numpy array input."""
        with pytest.raises(ValueError, match="query_vector must be a numpy.ndarray"):
            search._prepare_query_vector([1.0, 2.0, 3.0], 3)
    
    def test_invalid_dimensions(self):
        """Test error with multi-dimensional input."""
        vector = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 2D array
        
        with pytest.raises(ValueError, match="query_vector must be 1D"):
            search._prepare_query_vector(vector, 4)
    
    def test_dimension_mismatch(self):
        """Test error when vector size doesn't match expected dimension."""
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Dimensionality mismatch.*expects d=5.*got 3"):
            search._prepare_query_vector(vector, 5)
    
    def test_nan_values(self):
        """Test error with NaN values in vector."""
        vector = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        
        with pytest.raises(ValueError, match="query_vector contains NaN or Inf values"):
            search._prepare_query_vector(vector, 3)
    
    def test_inf_values(self):
        """Test error with infinite values in vector."""
        vector = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        
        with pytest.raises(ValueError, match="query_vector contains NaN or Inf values"):
            search._prepare_query_vector(vector, 3)


class TestDenseSearch:
    """Test suite for dense_search function."""
    
    def test_basic_search_success(self, sample_faiss_index, sample_query_vector):
        """Test basic successful dense search."""
        scores, indices = search.dense_search(
            query_vector=sample_query_vector,
            faiss_index=sample_faiss_index,
            top_k=3
        )
        
        assert len(scores) == 3
        assert len(indices) == 3
        assert len(scores) == len(indices)
        
        # Check data types
        assert all(isinstance(score, float) for score in scores)
        assert all(isinstance(idx, int) for idx in indices)
        
        # For IndexFlatIP, scores should be in descending order (higher = more similar)
        assert scores == sorted(scores, reverse=True)
        
        # Indices should be valid row numbers
        assert all(0 <= idx < sample_faiss_index.ntotal for idx in indices)
    
    def test_top_k_larger_than_index(self, sample_faiss_index, sample_query_vector):
        """Test when top_k is larger than the number of vectors in index."""
        index_size = sample_faiss_index.ntotal
        
        scores, indices = search.dense_search(
            query_vector=sample_query_vector,
            faiss_index=sample_faiss_index,
            top_k=index_size + 10  # Request more than available
        )
        
        # Should return all vectors in the index
        assert len(scores) == index_size
        assert len(indices) == index_size
    
    def test_top_k_zero_or_negative(self, sample_faiss_index, sample_query_vector):
        """Test error with invalid top_k values."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search.dense_search(sample_query_vector, sample_faiss_index, top_k=0)
        
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            search.dense_search(sample_query_vector, sample_faiss_index, top_k=-1)
    
    def test_empty_index(self, sample_query_vector):
        """Test behavior with empty FAISS index."""
        empty_index = faiss.IndexFlatIP(128)  # Empty index with 128 dimensions
        
        scores, indices = search.dense_search(
            query_vector=sample_query_vector,
            faiss_index=empty_index,
            top_k=5
        )
        
        assert scores == []
        assert indices == []
    
    def test_invalid_query_vector_dimensions(self, sample_faiss_index):
        """Test error with query vector of wrong dimensions."""
        wrong_dim_vector = np.array([1.0, 2.0], dtype=np.float32)  # 2D instead of 128D
        
        with pytest.raises(ValueError, match="Dimensionality mismatch"):
            search.dense_search(wrong_dim_vector, sample_faiss_index, top_k=3)
    
    def test_single_result(self, sample_faiss_index, sample_query_vector):
        """Test requesting only the top result."""
        scores, indices = search.dense_search(
            query_vector=sample_query_vector,
            faiss_index=sample_faiss_index,
            top_k=1
        )
        
        assert len(scores) == 1
        assert len(indices) == 1
        assert isinstance(scores[0], float)
        assert isinstance(indices[0], int)
    
    def test_cosine_similarity_behavior(self):
        """Test that IndexFlatIP with L2-normalized vectors gives cosine similarity."""
        # Create known vectors for predictable cosine similarity
        vectors = np.array([
            [1.0, 0.0, 0.0],  # Unit vector along x-axis
            [0.0, 1.0, 0.0],  # Unit vector along y-axis  
            [0.707, 0.707, 0.0],  # 45-degree angle from x-axis
        ], dtype=np.float32)
        
        index = faiss.IndexFlatIP(3)
        index.add(vectors)
        
        # Query with x-axis unit vector
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        scores, indices = search.dense_search(query, index, top_k=3)
        
        # Should return vectors in order of cosine similarity to [1,0,0]:
        # 1. [1,0,0] - cosine similarity = 1.0
        # 2. [0.707,0.707,0] - cosine similarity ≈ 0.707
        # 3. [0,1,0] - cosine similarity = 0.0
        
        assert indices[0] == 0  # First vector (identical)
        assert indices[1] == 2  # Third vector (45-degree)  
        assert indices[2] == 1  # Second vector (orthogonal)
        
        # Check approximate scores
        assert abs(scores[0] - 1.0) < 1e-6
        assert abs(scores[1] - 0.707) < 0.01
        assert abs(scores[2] - 0.0) < 1e-6


class TestKeywordSearch:
    """Test suite for keyword_search placeholder function."""
    
    def test_placeholder_returns_empty(self):
        """Test that keyword_search placeholder returns empty results."""
        mock_datastore = Mock()
        
        scores, indices = search.keyword_search(
            query="machine learning algorithms",
            datastore=mock_datastore,
            top_k=10
        )
        
        assert scores == []
        assert indices == []
    
    def test_placeholder_with_various_inputs(self):
        """Test placeholder behavior with different input types."""
        mock_datastore = Mock()
        
        # Different queries
        test_cases = [
            ("", 5),
            ("single word", 1),
            ("multiple word query with punctuation!", 20),
        ]
        
        for query, top_k in test_cases:
            scores, indices = search.keyword_search(query, mock_datastore, top_k)
            assert scores == []
            assert indices == []


class TestCombineResultsRRF:
    """Test suite for combine_results_rrf function."""
    
    def test_docstring_example(self):
        """Test the exact example from the docstring."""
        dense_indices = [101, 102, 103]
        sparse_indices = [102, 104, 101]
        
        combined_scores, combined_indices = search.combine_results_rrf(
            [(None, dense_indices), (None, sparse_indices)],
            k=60
        )
        
        # Verify the order matches expected RRF calculation
        # Actual order: [102, 101, 104, 103] based on RRF scores
        assert combined_indices == [102, 101, 104, 103]
        assert len(combined_scores) == 4
        
        # Verify scores are in descending order
        assert combined_scores == sorted(combined_scores, reverse=True)
        
        # Verify approximate RRF scores
        # doc 102: (1/(60+2)) + (1/(60+1)) = 0.0161 + 0.0164 = 0.0325
        # doc 101: (1/(60+1)) + (1/(60+3)) = 0.0164 + 0.0159 = 0.0323
        # doc 104: (1/(60+2)) = 0.0161
        # doc 103: (1/(60+3)) = 0.0159
        expected_102 = (1/(60+2)) + (1/(60+1))
        expected_101 = (1/(60+1)) + (1/(60+3))
        expected_104 = (1/(60+2))
        expected_103 = (1/(60+3))
        
        idx_102 = combined_indices.index(102)
        idx_101 = combined_indices.index(101)
        idx_104 = combined_indices.index(104)
        idx_103 = combined_indices.index(103)
        
        assert abs(combined_scores[idx_102] - expected_102) < 1e-6
        assert abs(combined_scores[idx_101] - expected_101) < 1e-6
        assert abs(combined_scores[idx_104] - expected_104) < 1e-6
        assert abs(combined_scores[idx_103] - expected_103) < 1e-6
    
    def test_single_result_list(self):
        """Test RRF with only one result list."""
        results = [([0.9, 0.8, 0.7], [10, 20, 30])]
        
        combined_scores, combined_indices = search.combine_results_rrf(results, k=60)
        
        # Should preserve original order since there's only one list
        assert combined_indices == [10, 20, 30]
        assert len(combined_scores) == 3
        
        # RRF scores should be: 1/61, 1/62, 1/63
        expected_scores = [1/(60+1), 1/(60+2), 1/(60+3)]
        for actual, expected in zip(combined_scores, expected_scores):
            assert abs(actual - expected) < 1e-6
    
    def test_empty_input(self):
        """Test RRF with empty input."""
        combined_scores, combined_indices = search.combine_results_rrf([], k=60)
        
        assert combined_scores == []
        assert combined_indices == []
    
    def test_empty_result_lists(self):
        """Test RRF with result lists that are all empty."""
        results = [([], []), ([], []), ([], [])]
        
        combined_scores, combined_indices = search.combine_results_rrf(results, k=60)
        
        assert combined_scores == []
        assert combined_indices == []
    
    def test_mixed_empty_and_non_empty(self):
        """Test RRF with mix of empty and non-empty result lists."""
        results = [
            ([], []),  # Empty
            ([0.9, 0.8], [100, 200]),  # Non-empty
            ([], []),  # Empty
            ([0.7, 0.6, 0.5], [200, 300, 400]),  # Non-empty, has overlap with previous
        ]
        
        combined_scores, combined_indices = search.combine_results_rrf(results, k=60)
        
        # Should have 4 unique documents: 100, 200, 300, 400
        # Document 200 appears in both non-empty lists
        assert len(combined_indices) == 4
        assert set(combined_indices) == {100, 200, 300, 400}
        
        # Document 200 should have highest score (appears twice)
        assert combined_indices[0] == 200
        
        # Verify RRF score for document 200: (1/(60+2)) + (1/(60+1))
        expected_200_score = (1/(60+2)) + (1/(60+1))
        assert abs(combined_scores[0] - expected_200_score) < 1e-6
    
    def test_no_overlap_between_lists(self):
        """Test RRF with completely disjoint result lists."""
        results = [
            ([0.9, 0.8], [10, 20]),
            ([0.7, 0.6], [30, 40]),
            ([0.5, 0.4], [50, 60]),
        ]
        
        combined_scores, combined_indices = search.combine_results_rrf(results, k=60)
        
        # Should have 6 unique documents
        assert len(combined_indices) == 6
        assert set(combined_indices) == {10, 20, 30, 40, 50, 60}
        
        # All documents should have same relative score pattern based on their ranks
        # Top-ranked docs from each list should score highest
        top_docs = {10, 30, 50}  # First in each list
        second_docs = {20, 40, 60}  # Second in each list
        
        # Find positions of top-ranked and second-ranked docs
        top_positions = [combined_indices.index(doc) for doc in top_docs if doc in combined_indices[:3]]
        second_positions = [combined_indices.index(doc) for doc in second_docs if doc in combined_indices[3:]]
        
        # Top-ranked docs should generally score higher than second-ranked docs
        assert len(top_positions) > 0
        assert len(second_positions) > 0
    
    def test_complete_overlap(self):
        """Test RRF with completely overlapping result lists."""
        indices = [100, 200, 300]
        results = [
            ([0.9, 0.8, 0.7], indices),
            ([0.95, 0.85, 0.75], indices),  # Same docs, different scores
        ]
        
        combined_scores, combined_indices = search.combine_results_rrf(results, k=60)
        
        # Should still have 3 documents but with boosted RRF scores
        assert len(combined_indices) == 3
        assert combined_indices == indices  # Same order since all docs appear in both lists
        
        # Each document should have RRF score = (1/(k+rank1)) + (1/(k+rank2))
        # Since both lists have same order, rank1 == rank2 for each doc
        for i, (score, doc_id) in enumerate(zip(combined_scores, combined_indices)):
            expected_score = 2 * (1/(60 + i + 1))  # Same rank in both lists
            assert abs(score - expected_score) < 1e-6
    
    def test_different_k_values(self):
        """Test RRF with different k parameter values."""
        results = [([0.9, 0.8], [10, 20]), ([0.7, 0.6], [20, 30])]
        
        # Test with k=60 (default)
        scores_60, indices_60 = search.combine_results_rrf(results, k=60)
        
        # Test with k=1 (more emphasis on rank differences)
        scores_1, indices_1 = search.combine_results_rrf(results, k=1)
        
        # Test with k=1000 (less emphasis on rank differences)
        scores_1000, indices_1000 = search.combine_results_rrf(results, k=1000)
        
        # Order should be same (doc 20 appears in both lists)
        assert indices_60[0] == indices_1[0] == indices_1000[0] == 20
        
        # But scores should be different
        assert scores_60[0] != scores_1[0] != scores_1000[0]
        
        # Verify actual RRF calculations for document 20
        # k=60: (1/(60+2)) + (1/(60+1)) 
        # k=1: (1/(1+2)) + (1/(1+1))
        # k=1000: (1/(1000+2)) + (1/(1000+1))
        
        expected_60 = (1/(60+2)) + (1/(60+1))
        expected_1 = (1/(1+2)) + (1/(1+1))
        expected_1000 = (1/(1000+2)) + (1/(1000+1))
        
        assert abs(scores_60[0] - expected_60) < 1e-6
        assert abs(scores_1[0] - expected_1) < 1e-6
        assert abs(scores_1000[0] - expected_1000) < 1e-6
    
    def test_large_result_sets(self):
        """Test RRF with larger result sets to verify performance and correctness."""
        # Create two large overlapping result lists
        list1_indices = list(range(0, 1000, 2))  # Even numbers 0-998
        list2_indices = list(range(1, 1000, 2))  # Odd numbers 1-999
        overlap_indices = list(range(500, 600))  # Some overlap
        
        results = [
            ([0.0] * len(list1_indices + overlap_indices), list1_indices + overlap_indices),
            ([0.0] * len(list2_indices + overlap_indices), list2_indices + overlap_indices),
        ]
        
        combined_scores, combined_indices = search.combine_results_rrf(results, k=60)
        
        # Should have all unique indices
        expected_unique = set(list1_indices + list2_indices + overlap_indices)
        assert set(combined_indices) == expected_unique
        
        # Overlapping docs should score highest
        overlap_positions = [combined_indices.index(idx) for idx in overlap_indices[:10]]
        # Most overlap docs should be in top portion of results
        assert sum(1 for pos in overlap_positions if pos < len(combined_indices) // 3) > 5
    
    def test_score_ordering_consistency(self):
        """Test that RRF scores are always in descending order."""
        # Test with various configurations
        test_cases = [
            [([0.9, 0.8, 0.7], [1, 2, 3])],
            [([0.9, 0.8], [1, 2]), ([0.7, 0.6], [2, 3])],
            [([0.9], [1]), ([0.8], [2]), ([0.7], [3])],
            [([0.9, 0.8, 0.7, 0.6, 0.5], [1, 2, 3, 4, 5]), ([0.4, 0.3, 0.2], [3, 4, 6])],
        ]
        
        for results in test_cases:
            combined_scores, combined_indices = search.combine_results_rrf(results, k=60)
            
            # Scores should always be in descending order
            assert combined_scores == sorted(combined_scores, reverse=True), f"Scores not ordered for {results}"
            
            # Should have same length
            assert len(combined_scores) == len(combined_indices)
            
            # All scores should be positive (since RRF formula gives positive values)
            assert all(score > 0 for score in combined_scores)


class TestSearchModuleIntegration:
    """Integration tests combining multiple search functions."""
    
    def test_dense_to_rrf_pipeline(self, sample_faiss_index, sample_vectors):
        """Test pipeline from dense search to RRF combination."""
        # Create two different queries
        query1 = sample_vectors[0] * 0.9 + np.random.randn(128).astype(np.float32) * 0.1
        query2 = sample_vectors[1] * 0.9 + np.random.randn(128).astype(np.float32) * 0.1
        query1 = query1 / np.linalg.norm(query1)  # Normalize
        query2 = query2 / np.linalg.norm(query2)  # Normalize
        
        # Run dense searches
        dense_results1 = search.dense_search(query1, sample_faiss_index, top_k=3)
        dense_results2 = search.dense_search(query2, sample_faiss_index, top_k=3)
        
        # Combine with RRF
        combined_scores, combined_indices = search.combine_results_rrf([
            dense_results1,
            dense_results2
        ], k=60)
        
        # Verify results
        assert len(combined_indices) <= 6  # At most 6 unique indices (could be fewer due to overlap)
        assert len(combined_scores) == len(combined_indices)
        assert combined_scores == sorted(combined_scores, reverse=True)
        
        # All indices should be valid
        assert all(0 <= idx < sample_faiss_index.ntotal for idx in combined_indices)
    
    def test_dense_plus_keyword_placeholder_pipeline(self, sample_faiss_index, sample_query_vector):
        """Test combining dense search with keyword search placeholder."""
        mock_datastore = Mock()
        
        # Get dense results
        dense_results = search.dense_search(sample_query_vector, sample_faiss_index, top_k=3)
        
        # Get keyword results (will be empty)
        keyword_results = search.keyword_search("test query", mock_datastore, top_k=5)
        
        # Combine with RRF
        combined_scores, combined_indices = search.combine_results_rrf([
            dense_results,
            keyword_results
        ], k=60)
        
        # Should equal dense results since keyword is empty
        assert combined_indices == dense_results[1]
        
        # Scores should be RRF-transformed versions of ranks
        expected_rrf_scores = [1/(60+i+1) for i in range(len(dense_results[1]))]
        for actual, expected in zip(combined_scores, expected_rrf_scores):
            assert abs(actual - expected) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
