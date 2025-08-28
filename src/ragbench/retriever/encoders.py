from __future__ import annotations

"""EncoderManager: unified model manager for embedding & reranking.

Public API (contract):
    - encode_query(query: str) -> np.ndarray
    - rerank(query: str, chunks: List[dict]) -> List[dict]

Notes
-----
* Embeddings: BGE family bi‑encoder (default: BAAI/bge-large-en-v1.5).
* Reranker: Cross‑encoder (default: BAAI/bge-reranker-large).
* Apple Silicon: Uses torch MPS if available. Falls back to CUDA, then CPU.
"""

from typing import List, Dict, Optional
import logging

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

logger = logging.getLogger(__name__)


def _select_device(preferred: Optional[str] = None) -> torch.device:
    """Select the best available torch device with Apple Silicon support.
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2)
    if norm < eps:
        return x
    return x / norm


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using attention mask.

    Args:
        last_hidden_state: [B, T, H]
        attention_mask:    [B, T]
    Returns:
        [B, H] pooled embeddings
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = (last_hidden_state * input_mask_expanded).sum(dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class EncoderManager:
    """Central manager for embedding and reranking models.

    Parameters
    ----------
    embed_model_name : str
        Hugging Face model id for the bi-encoder (sentence embedding) model.
    reranker_model_name : str
        Hugging Face model id for the cross-encoder reranker model.
    device : Optional[str]
        Explicit torch device string (e.g., "mps", "cuda", "cpu"). If None, auto-select.
    embedding_max_length : int
        Max sequence length for the embedding tokenizer.
    rerank_max_length : int
        Max combined length for cross-encoder inputs.
    rerank_batch_size : int
        Number of (query, passage) pairs per forward pass during reranking.
    text_field : str
        Dictionary key in each chunk containing the text to score (default: "embedding_text").
    """

    def __init__(
        self,
        embed_model_name: str = "BAAI/bge-large-en-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-large",
        device: Optional[str] = None,
        embedding_max_length: int = 512,
        rerank_max_length: int = 512,
        rerank_batch_size: int = 32,
        text_field: str = "embedding_text",
    ) -> None:
        self.device: torch.device = _select_device(device)

        # --- Embedding model (bi-encoder) ---
        self._embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self._embed_model = AutoModel.from_pretrained(embed_model_name)
        self._embed_model.eval()
        self._embed_model.to(self.device)
        self.embedding_max_length: int = embedding_max_length

        # --- Reranker (cross-encoder) ---
        self._rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self._rerank_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
        self._rerank_model.eval()
        self._rerank_model.to(self.device)
        self.rerank_max_length: int = rerank_max_length
        self.rerank_batch_size: int = rerank_batch_size

        # --- Misc config ---
        self.text_field: str = text_field

        logger.info(
            "EncoderManager initialized on device=%s | embed=%s | reranker=%s",
            self.device,
            embed_model_name,
            reranker_model_name,
        )

    # -------------------- Public API --------------------
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query string into a normalized embedding vector.

        Adds the BGE "query: " prefix, tokenizes, mean‑pools, and L2‑normalizes.

        Parameters
        ----------
        query : str
            The user query to encode.

        Returns
        -------
        np.ndarray
            L2‑normalized embedding vector with shape [hidden_dim].
        """
        prefixed = f"query: {query}".strip()
        with torch.no_grad():
            inputs = self._embed_tokenizer(
                prefixed,
                max_length=self.embedding_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._embed_model(**inputs)
            pooled = _mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])  # [1, H]
            vec = pooled.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return _l2_normalize(vec)

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Score and sort chunks by cross-encoder relevance to the query.

        The method computes a relevance score for each chunk's `text_field`
        against the given query, using a cross-encoder. It returns a new list
        where each chunk dict is augmented with a `rerank_score` field and the
        list is sorted by that score in descending order.

        Parameters
        ----------
        query : str
            The user query to score against.
        chunks : List[dict]
            A list of chunk dictionaries; each must contain `self.text_field`.

        Returns
        -------
        List[dict]
            Chunks sorted by descending relevance score; each includes
            `rerank_score` (float).
        """
        if not chunks:
            return []

        # Prepare texts; fall back to 'text' if embedding_text is missing, but log it.
        texts: List[str] = []
        fallback_count = 0
        for ch in chunks:
            t = ch.get(self.text_field)
            if t is None:
                t = ch.get("text")
                fallback_count += 1
            if t is None:
                raise ValueError(
                    f"Chunk {ch.get('chunk_id', '<unknown>')} lacks both '{self.text_field}' and 'text' fields"
                )
            texts.append(str(t))
        if fallback_count:
            logger.warning(
                "rerank: %d/%d chunks missing '%s' (fell back to 'text').",
                fallback_count,
                len(chunks),
                self.text_field,
            )

        # Batch the (query, text) pairs for cross-encoder scoring.
        scores: List[float] = []
        self._rerank_model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), self.rerank_batch_size):
                batch_texts = texts[start : start + self.rerank_batch_size]
                pairs = [(query, passage) for passage in batch_texts]
                enc = self._rerank_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.rerank_max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self._rerank_model(**enc)
                # Many rerankers output logits of shape [B, 1]; squeeze safely.
                logits = out.logits.squeeze(-1).detach().cpu().float().numpy()
                scores.extend(logits.tolist())

        # Attach scores and sort (descending)
        enriched: List[Dict] = []
        for ch, s in zip(chunks, scores):
            new_ch = dict(ch)
            new_ch["rerank_score"] = float(s)
            enriched.append(new_ch)

        enriched.sort(key=lambda d: d["rerank_score"], reverse=True)
        return enriched
