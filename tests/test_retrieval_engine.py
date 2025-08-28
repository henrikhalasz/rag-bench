"""
Comprehensive test suite for the RetrievalEngine class.

This test suite creates temporary FAISS indexes and data artifacts for testing,
ensuring the complete retrieval pipeline works correctly.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import faiss

# Ensure the src directory is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ragbench.retriever.datastore import DataStore
from ragbench.retriever.encoders import EncoderManager
from ragbench.retriever.engine import RetrievalEngine
from ragbench.retriever.search import dense_search, keyword_search, combine_results_rrf


@pytest.fixture
def sample_retrieval_data() -> List[Dict]:
    """Create sample data specifically designed for retrieval testing."""
    return [
        {
            "chunk_id": "doc_001",
            "text": "Machine learning algorithms for classification and regression tasks.",
            "embedding_text": "machine learning classification regression algorithms",
            "prev_id": None,
            "next_id": "doc_002",
            "section_path": "ml_basics",
            "page": 1,
            "token_count": 55,
        },
        {
            "chunk_id": "doc_002",
            "text": "Neural networks with backpropagation training methodology.",
            "embedding_text": "neural networks backpropagation training methodology",
            "prev_id": "doc_001",
            "next_id": "doc_003",
            "section_path": "ml_basics",
            "page": 1,
            "token_count": 48,
        },
        {
            "chunk_id": "doc_003",
            "text": "Deep learning architectures including convolutional and recurrent networks.",
            "embedding_text": "deep learning convolutional recurrent neural networks",
            "prev_id": "doc_002",
            "next_id": "doc_004",
            "section_path": "deep_learning",
            "page": 2,
            "token_count": 62,
        },
        {
            "chunk_id": "doc_004",
            "text": "Natural language processing with transformer models and attention.",
            "embedding_text": "natural language processing transformers attention mechanisms",
            "prev_id": "doc_003",
            "next_id": "doc_005",
            "section_path": "nlp",
            "page": 3,
            "token_count": 51,
        },
        {
            "chunk_id": "doc_005",
            "text": "Computer vision applications using convolutional neural networks.",
            "embedding_text": "computer vision convolutional neural networks applications",
            "prev_id": "doc_004",
            "next_id": None,
            "section_path": "computer_vision",
            "page": 4,
            "token_count": 47,
        },
    ]


@pytest.fixture
def temp_retrieval_datastore(sample_retrieval_data):
    """Create a temporary DataStore for retrieval testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        parquet_path = Path(temp_dir) / "retrieval_chunks.parquet"
        df = pd.DataFrame(sample_retrieval_data)
        df.to_parquet(parquet_path, index=False)
        yield DataStore(str(parquet_path))


@pytest.fixture
def temp_faiss_index(sample_retrieval_data):
    """Create a temporary FAISS index with synthetic embeddings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create synthetic embeddings that reflect the content
        # Each vector has 384 dimensions (matching all-MiniLM-L6-v2)
        embeddings = []
        np.random.seed(42)  # For reproducible tests
        
        for i, chunk in enumerate(sample_retrieval_data):
            # Create embeddings that cluster by topic
            base_vector = np.random.normal(0, 0.1, 384).astype(np.float32)
            
            # Add topic-specific signal
            if "machine learning" in chunk["embedding_text"] or "neural" in chunk["embedding_text"]:
                base_vector[:10] += 0.5  # ML signal
            if "deep learning" in chunk["embedding_text"] or "convolutional" in chunk["embedding_text"]:
                base_vector[10:20] += 0.5  # Deep learning signal
            if "language processing" in chunk["embedding_text"] or "transformer" in chunk["embedding_text"]:
                base_vector[20:30] += 0.5  # NLP signal
            if "computer vision" in chunk["embedding_text"]:
                base_vector[30:40] += 0.5  # Vision signal
                
            # L2 normalize
            norm = np.linalg.norm(base_vector)
            if norm > 0:
                base_vector = base_vector / norm
                
            embeddings.append(base_vector)
        
        # Create FAISS index
        embeddings_array = np.vstack(embeddings)
        d = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        index.add(embeddings_array)
        
        # Save to temporary file
        index_path = Path(temp_dir) / "test_index.faiss"
        faiss.write_index(index, str(index_path))
        
        yield str(index_path)


@pytest.fixture(scope="session")
def mock_encoder_manager():
    """Create a mocked EncoderManager for fast testing."""
    mock_encoder = MagicMock(spec=EncoderManager)
    
    # Add required attributes that match the real implementation
    mock_encoder.embedding_max_length = 128
    mock_encoder.rerank_max_length = 128
    mock_encoder.rerank_batch_size = 4
    mock_encoder.text_field = "embedding_text"
    mock_encoder.device = "cpu"
    mock_encoder.embedding_dim = 384
    
    # Mock encode_query to return a predictable vector
    def mock_encode_query(query: str) -> np.ndarray:
        # Generate different vectors for different query types
        np.random.seed(hash(query) % 2**31)  # Deterministic but query-dependent
        vector = np.random.normal(0, 0.1, 384).astype(np.float32)
        
        # Add query-specific signals
        if "machine learning" in query.lower() or "classification" in query.lower():
            vector[:10] += 0.5
        elif "deep learning" in query.lower() or "neural" in query.lower():
            vector[10:20] += 0.5
        elif "nlp" in query.lower() or "language" in query.lower():
            vector[20:30] += 0.5
        elif "vision" in query.lower() or "computer" in query.lower():
            vector[30:40] += 0.5
            
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    # Mock rerank to sort by simple text similarity
    def mock_rerank(query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []
            
        query_lower = query.lower()
        
        # Add rerank scores based on simple keyword matching
        scored_chunks = []
        for chunk in chunks:
            chunk_copy = dict(chunk)
            
            # Validate required fields like the real implementation
            text_field = mock_encoder.text_field
            if text_field not in chunk_copy and "text" not in chunk_copy:
                raise ValueError(f"Chunk {chunk_copy.get('chunk_id', '<unknown>')} lacks both '{text_field}' and 'text' fields")
            
            text = chunk.get("embedding_text", "").lower()
            
            # Simple scoring based on keyword overlap
            query_words = set(query_lower.split())
            text_words = set(text.split())
            overlap = len(query_words.intersection(text_words))
            
            # Base score plus overlap bonus
            base_score = 2.0
            overlap_bonus = overlap * 0.5
            chunk_copy["rerank_score"] = base_score + overlap_bonus
            scored_chunks.append(chunk_copy)
        
        # Sort by rerank_score descending
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_chunks
    
    mock_encoder.encode_query = mock_encode_query
    mock_encoder.rerank = mock_rerank
    
    return mock_encoder


@pytest.fixture
def engine(temp_retrieval_datastore, temp_faiss_index, mock_encoder_manager):
    """Create a RetrievalEngine instance for testing."""
    # Load the FAISS index from the path
    faiss_index = faiss.read_index(temp_faiss_index)
    
    # Create engine with injected dependencies for testing
    return RetrievalEngine(
        datastore=temp_retrieval_datastore,
        faiss_index=faiss_index,
        encoder_manager=mock_encoder_manager,
    )


class TestSearchFunctions:
    """Test the search utility functions used by the engine."""
    
    def test_dense_search_success(self, temp_faiss_index):
        """Test successful dense search operation."""
        # Load the test index
        index = faiss.read_index(temp_faiss_index)
        
        # Create a query vector similar to one in the index
        np.random.seed(42)
        query_vector = np.random.normal(0, 0.1, 384).astype(np.float32)
        query_vector[:10] += 0.5  # ML signal to match first chunk
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        scores, indices = dense_search(query_vector, index, top_k=3)
        
        assert len(scores) == 3
        assert len(indices) == 3
        assert all(isinstance(s, float) for s in scores)
        assert all(isinstance(i, int) for i in indices)
        assert all(0 <= i < index.ntotal for i in indices)
        
        # Scores should be in descending order for IP index
        assert scores[0] >= scores[1] >= scores[2]
    
    def test_dense_search_empty_index(self):
        """Test dense search with empty index."""
        empty_index = faiss.IndexFlatIP(384)
        query_vector = np.random.normal(0, 1, 384).astype(np.float32)
        
        scores, indices = dense_search(query_vector, empty_index, top_k=5)
        
        assert scores == []
        assert indices == []
    
    def test_dense_search_invalid_inputs(self, temp_faiss_index):
        """Test dense search with invalid inputs."""
        index = faiss.read_index(temp_faiss_index)
        
        # Test invalid top_k
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            dense_search(np.random.normal(0, 1, 384).astype(np.float32), index, top_k=0)
        
        # Test wrong dimension
        with pytest.raises(ValueError, match="Dimensionality mismatch"):
            dense_search(np.random.normal(0, 1, 100).astype(np.float32), index, top_k=1)
        
        # Test non-1D vector
        with pytest.raises(ValueError, match="must be 1D"):
            dense_search(np.random.normal(0, 1, (2, 384)).astype(np.float32), index, top_k=1)
    
    def test_keyword_search_placeholder(self, temp_retrieval_datastore):
        """Test the keyword search placeholder functionality."""
        scores, indices = keyword_search("test query", temp_retrieval_datastore, top_k=5)
        
        # Should return empty results (placeholder)
        assert scores == []
        assert indices == []
    
    def test_combine_results_rrf_basic(self):
        """Test basic RRF combination functionality."""
        # Test data: two ranked lists
        dense_results = ([0.9, 0.8, 0.7], [101, 102, 103])
        sparse_results = ([2.0, 1.5, 1.0], [102, 104, 101])
        
        combined_scores, combined_indices = combine_results_rrf(
            [dense_results, sparse_results], k=60
        )
        
        assert len(combined_scores) == 4  # Unique documents
        assert len(combined_indices) == 4
        
        # doc 102 should rank highest (appears in both lists early)
        assert combined_indices[0] == 102
        
        # All scores should be positive
        assert all(score > 0 for score in combined_scores)
    
    def test_combine_results_rrf_empty(self):
        """Test RRF with empty inputs."""
        # Empty results list
        scores, indices = combine_results_rrf([])
        assert scores == []
        assert indices == []
        
        # Results with empty lists
        scores, indices = combine_results_rrf([([], []), ([], [])])
        assert scores == []
        assert indices == []
    
    def test_combine_results_rrf_single_list(self):
        """Test RRF with a single result list."""
        single_results = ([1.0, 0.9, 0.8], [100, 101, 102])
        
        combined_scores, combined_indices = combine_results_rrf([single_results])
        
        assert len(combined_indices) == 3
        assert combined_indices == [100, 101, 102]  # Should preserve order


class TestRetrievalEngine:
    """Test the main RetrievalEngine class."""
    
    @pytest.fixture
    def engine(self, temp_retrieval_datastore, temp_faiss_index, mock_encoder_manager):
        """Create a RetrievalEngine instance for testing."""
        # Load the FAISS index from the path
        faiss_index = faiss.read_index(temp_faiss_index)
        
        # Create engine with injected dependencies for testing
        return RetrievalEngine(
            datastore=temp_retrieval_datastore,
            faiss_index=faiss_index,
            encoder_manager=mock_encoder_manager,
        )
    
    def test_initialization_success(self, engine):
        """Test successful engine initialization."""
        assert engine is not None
        assert engine._datastore is not None
        assert engine._faiss_index is not None
        assert engine._encoders is not None
        assert len(engine._row_idx_to_chunk_id) == 5
    
    def test_initialization_parameter_validation(self, temp_retrieval_datastore):
        """Test constructor parameter validation."""
        # Test missing both datastore_path and datastore
        with pytest.raises(ValueError, match="Either datastore_path or datastore must be provided"):
            RetrievalEngine()
        
        # Test missing both faiss_index_path and faiss_index
        with pytest.raises(ValueError, match="Either faiss_index_path or faiss_index must be provided"):
            RetrievalEngine(datastore_path=temp_retrieval_datastore._parquet_path)
    
    def test_initialization_with_file_paths(self, temp_retrieval_datastore, temp_faiss_index):
        """Test backward compatibility with file path based initialization."""
        # Should work with the original file-path based constructor
        engine = RetrievalEngine(
            datastore_path=temp_retrieval_datastore._parquet_path,
            faiss_index_path=temp_faiss_index,
            embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
            reranker_model_name="cross-encoder/ms-marco-MiniLM-L-2-v2",
        )
        
        assert engine is not None
        assert engine._datastore is not None
        assert engine._faiss_index is not None
        assert engine._encoders is not None
    
    def test_initialization_with_mismatched_sizes(self, temp_retrieval_datastore, mock_encoder_manager):
        """Test initialization warning when FAISS and datastore sizes don't match."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a smaller FAISS index (2 vectors instead of 5)
            small_embeddings = np.random.normal(0, 1, (2, 384)).astype(np.float32)
            small_index = faiss.IndexFlatIP(384)
            small_index.add(small_embeddings)
            
            with patch('ragbench.retriever.engine.logger') as mock_logger:
                engine = RetrievalEngine(
                    datastore=temp_retrieval_datastore,
                    faiss_index=small_index,
                    encoder_manager=mock_encoder_manager,
                )
                
                # Should log a warning about size mismatch
                mock_logger.warning.assert_called_once()
                assert "FAISS ntotal" in str(mock_logger.warning.call_args)
    
    def test_retrieve_basic_functionality(self, engine):
        """Test basic retrieve functionality."""
        results = engine.retrieve(
            "machine learning classification",
            top_k_dense=3,
            top_k_sparse=3,
            top_k_rerank=2,
            context_window_size=1,
        )
        
        assert isinstance(results, list)
        assert len(results) <= 2  # top_k_rerank=2
        
        # Check result structure
        for result in results:
            assert "chunk" in result
            assert "rerank_score" in result
            assert "neighbors" in result
            
            chunk = result["chunk"]
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "rerank_score" in chunk
            
            neighbors = result["neighbors"]
            assert "prev" in neighbors
            assert "next" in neighbors
            assert isinstance(neighbors["prev"], list)
            assert isinstance(neighbors["next"], list)
    
    def test_retrieve_empty_query(self, engine):
        """Test retrieve with empty query."""
        results = engine.retrieve("")
        assert results == []
        
        results = engine.retrieve("   ")  # whitespace only
        assert results == []
    
    def test_retrieve_parameter_validation(self, engine):
        """Test parameter validation in retrieve method."""
        # Test invalid top_k values
        with pytest.raises(ValueError, match="top_k_dense.*must be positive"):
            engine.retrieve("test", top_k_dense=0)
        
        with pytest.raises(ValueError, match="top_k_sparse.*must be positive"):
            engine.retrieve("test", top_k_sparse=-1)
        
        with pytest.raises(ValueError, match="top_k_rerank.*must be.*non-negative"):
            engine.retrieve("test", top_k_rerank=-1)
        
        with pytest.raises(ValueError, match="context_window_size.*must be.*non-negative"):
            engine.retrieve("test", context_window_size=-1)
        
        with pytest.raises(ValueError, match="rrf_k must be a positive integer"):
            engine.retrieve("test", rrf_k=0)
    
    def test_retrieve_with_explicit_candidate_cap(self, engine):
        """Test retrieve with explicit rerank candidate cap."""
        results = engine.retrieve(
            "neural networks deep learning",
            top_k_dense=5,
            top_k_sparse=5,
            top_k_rerank=2,
            rerank_candidate_cap=3,
        )
        
        assert len(results) <= 2  # Still limited by top_k_rerank
    
    def test_retrieve_with_invalid_candidate_cap(self, engine):
        """Test retrieve with invalid candidate cap."""
        with pytest.raises(ValueError, match="rerank_candidate_cap must be a positive integer"):
            engine.retrieve("test", rerank_candidate_cap=0)
    
    def test_retrieve_zero_top_k_rerank(self, engine):
        """Test retrieve with top_k_rerank=0."""
        results = engine.retrieve(
            "machine learning",
            top_k_rerank=0,
        )
        
        assert results == []
    
    def test_retrieve_large_context_window(self, engine):
        """Test retrieve with large context window."""
        results = engine.retrieve(
            "neural networks",
            top_k_rerank=1,
            context_window_size=10,  # Larger than available chain
        )
        
        assert len(results) == 1
        # Should still work, just get all available neighbors
        neighbors = results[0]["neighbors"]
        assert isinstance(neighbors["prev"], list)
        assert isinstance(neighbors["next"], list)
    
    def test_retrieve_with_neighbor_lookup_failure(self, engine):
        """Test retrieve when neighbor lookup fails."""
        # Mock the datastore to raise an exception during neighbor lookup
        original_get_neighbors = engine._datastore.get_neighbors
        
        def failing_get_neighbors(chunk_id, window_size):
            raise RuntimeError("Simulated neighbor lookup failure")
        
        engine._datastore.get_neighbors = failing_get_neighbors
        
        try:
            results = engine.retrieve(
                "machine learning",
                top_k_rerank=1,
                context_window_size=1,
            )
            
            # Should still return results but with empty neighbors
            assert len(results) == 1
            assert results[0]["neighbors"] == {"prev": [], "next": []}
        finally:
            # Restore original method
            engine._datastore.get_neighbors = original_get_neighbors
    
    def test_retrieve_performance_characteristics(self, engine):
        """Test that retrieve completes within reasonable time bounds."""
        import time
        
        start_time = time.time()
        results = engine.retrieve(
            "machine learning classification",
            top_k_rerank=2,
        )
        elapsed_time = time.time() - start_time
        
        # Should complete quickly with mock components (within 5 seconds)
        assert elapsed_time < 5.0
        
        # Should return meaningful results
        assert len(results) > 0
        assert len(results) <= 2  # Respects top_k_rerank
        
        # Results should have expected structure
        for result in results:
            assert "chunk" in result
            assert "rerank_score" in result
            assert "neighbors" in result


class TestRetrievalEngineIntegration:
    """Integration tests using real data files when available."""
    
    def test_with_real_data_if_available(self):
        """Test with real data files if they exist."""
        # Check if real data files exist
        data_dir = Path(__file__).parent.parent / "data" / "processed"
        embeddings_dir = data_dir / "embeddings"
        
        faiss_files = list(embeddings_dir.glob("*.faiss"))
        parquet_files = list(data_dir.glob("*_chunks.jsonl"))  # Note: might need conversion to parquet
        
        if not faiss_files:
            pytest.skip("No FAISS index files found for integration testing")
        
        if not parquet_files:
            pytest.skip("No chunk files found for integration testing")
        
        # If we have the data, we could run a real test here
        # For now, just verify the files exist
        assert len(faiss_files) > 0
        assert len(parquet_files) > 0
    
    def test_query_relevance_with_mock_data(self, engine):
        """Test that query results are relevant using mock data."""
        # Test specific domain queries
        ml_results = engine.retrieve(
            "machine learning classification algorithms",
            top_k_rerank=2,
        )
        
        # Should return results
        assert len(ml_results) > 0
        
        # Check that the top result is relevant
        top_chunk = ml_results[0]["chunk"]
        top_text = top_chunk.get("embedding_text", "").lower()
        
        # Should contain relevant terms (based on our mock reranker)
        relevant_terms = ["machine", "learning", "classification", "algorithms"]
        found_terms = sum(1 for term in relevant_terms if term in top_text)
        assert found_terms > 0  # At least one relevant term should be found
    
    def test_context_expansion_quality(self, engine):
        """Test that context expansion provides meaningful neighbors."""
        results = engine.retrieve(
            "neural networks",
            top_k_rerank=1,
            context_window_size=2,
        )
        
        assert len(results) == 1
        
        neighbors = results[0]["neighbors"]
        
        # Should have both prev and next keys
        assert "prev" in neighbors
        assert "next" in neighbors
        
        # For the middle documents, should have both prev and next neighbors
        all_neighbors = neighbors["prev"] + neighbors["next"]
        
        # Each neighbor should have the required chunk structure
        for neighbor in all_neighbors:
            assert "chunk_id" in neighbor
            assert "text" in neighbor
            assert "embedding_text" in neighbor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
