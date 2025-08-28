from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataStore:
    """Parquet-backed metadata store for RAG retrieval.

    This class memory-maps the backing Parquet file for fast random row access
    and builds three explicit lookup maps for O(1) resolution of chunk metadata.

    Design goals:
    - Use `pyarrow.dataset` to interface with the Parquet source (no pandas at init).
    - Keep a memory-mapped Arrow Table for efficient `take()` on row indices.
    - Provide transparent, deterministic lookups with explicit validations.

    Required Parquet columns (schema contract):
        chunk_id: string (unique primary key)
        prev_id:  string or null
        next_id:  string or null
        section_path: string
        ... other columns are passed through untouched and returned to the caller.
    """

    # ---------- Types ----------
    _IdxMap = Dict[str, int]
    _NeighborMap = Dict[str, Tuple[Optional[str], Optional[str]]]
    _SectionMap = Dict[str, List[str]]

    # ---------- Init ----------
    def __init__(self, parquet_path: str) -> None:
        """Initialize the DataStore.

        Args:
            parquet_path: Path to the Parquet file containing metadata.
        """
        self._parquet_path: str = parquet_path

        # Expose the dataset handle (useful for external scans/inspections).
        # Using pyarrow.dataset to satisfy the loading requirement and to
        # future‑proof for partitioned datasets.
        self._dataset: ds.Dataset = ds.dataset(parquet_path, format="parquet")

        # Memory-map the full table for zero-copy random access with take().
        # This way, we can work with datasets larger than our available RAM.
        self._table: pa.Table = pq.read_table(parquet_path, memory_map=True)

        # Validate required columns exist.
        required_cols = {"chunk_id", "prev_id", "next_id", "section_path"}
        missing = required_cols.difference(set(self._table.schema.names))
        if missing:
            raise ValueError(
                f"DataStore initialization failed: missing required columns {sorted(missing)}"
            )

        # Build O(1) lookup maps using the same table we will take() from,
        # ensuring index alignment.
        self._chunk_id_to_row_idx: DataStore._IdxMap = {}
        self._chunk_id_to_neighbors: DataStore._NeighborMap = {}
        self._section_to_chunk_ids: DataStore._SectionMap = {}

        # Convert entire columns at once for better performance
        chunk_ids = self._table["chunk_id"].to_pylist()
        prev_ids = self._table["prev_id"].to_pylist()
        next_ids = self._table["next_id"].to_pylist()
        sections = self._table["section_path"].to_pylist()

        # Build maps in batch
        for i, (chunk_id, prev_id, next_id, section) in enumerate(zip(chunk_ids, prev_ids, next_ids, sections)):
            if chunk_id in self._chunk_id_to_row_idx:
                raise ValueError(f"Duplicate chunk_id detected: {chunk_id}")

            self._chunk_id_to_row_idx[chunk_id] = i
            self._chunk_id_to_neighbors[chunk_id] = (prev_id, next_id)

            if section is None:
                section = ""
            self._section_to_chunk_ids.setdefault(section, []).append(chunk_id)

        # Perform integrity checks on neighbor pointers.
        self._validate_data()

        logger.info(
            "DataStore initialized: %s rows, %s sections",
            self._table.num_rows,
            len(self._section_to_chunk_ids),
        )

    # ---------- Private helpers ----------
    def _validate_data(self) -> None:
        """Validate referential integrity of neighbor pointers.

        Ensures that each non-null prev_id/next_id references a valid chunk_id.
        Raises:
            ValueError: If dangling neighbor references are found.
        """
        missing_refs: List[Tuple[str, str]] = []  # (from_chunk_id, missing_ref)

        for cid, (prev_id, next_id) in self._chunk_id_to_neighbors.items():
            if prev_id is not None and prev_id not in self._chunk_id_to_row_idx:
                missing_refs.append((cid, prev_id))
            if next_id is not None and next_id not in self._chunk_id_to_row_idx:
                missing_refs.append((cid, next_id))

        if missing_refs:
            # Build a compact error message with a sample of missing refs
            sample = ", ".join(
                f"{src}->{ref}" for src, ref in missing_refs[:10]
            )
            msg = (
                f"Neighbor integrity check failed: {len(missing_refs)} dangling references. "
                f"Examples: {sample}"
            )
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Neighbor integrity check passed: 0 dangling references.")

    # ---------- Public API ----------
    @lru_cache(maxsize=1000)
    def _fetch_chunks_by_ids_cached(self, chunk_ids_tuple: Tuple[str, ...]) -> List[dict]:
        """Cached version of fetch_chunks_by_ids for tuple input."""
        chunk_ids = list(chunk_ids_tuple)
        if not chunk_ids:
            return []

        missing = [cid for cid in chunk_ids if cid not in self._chunk_id_to_row_idx]
        if missing:
            raise ValueError(
                f"fetch_chunks_by_ids: unknown chunk_id(s): {missing[:10]}"
                + ("..." if len(missing) > 10 else "")
            )

        indices = [self._chunk_id_to_row_idx[cid] for cid in chunk_ids]
        idx_array = pa.array(indices, type=pa.int64())
        taken: pa.Table = self._table.take(idx_array)
        return taken.to_pylist()

    def fetch_chunks_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """Fetch rows for the specified chunk_ids using Arrow's take().

        Args:
            chunk_ids: List of chunk_id strings to fetch.

        Returns:
            A list of Python dictionaries, each representing a row.

        Raises:
            ValueError: If any requested chunk_id is not present in the store.
        """
        # Convert to tuple for caching
        return self._fetch_chunks_by_ids_cached(tuple(chunk_ids))

    def fetch_chunks_by_ids_partial(self, chunk_ids: List[str]) -> Tuple[List[dict], List[str]]:
        """Fetch rows for chunk_ids, returning both found chunks and missing IDs.

        Args:
            chunk_ids: List of chunk_id strings to fetch.

        Returns:
            A tuple of (found_chunks, missing_chunk_ids).
            found_chunks: List of dictionaries for chunks that were found.
            missing_chunk_ids: List of chunk_ids that were not found in the store.
        """
        if not chunk_ids:
            return [], []

        found_ids = []
        missing_ids = []
        
        for cid in chunk_ids:
            if cid in self._chunk_id_to_row_idx:
                found_ids.append(cid)
            else:
                missing_ids.append(cid)

        if not found_ids:
            return [], missing_ids

        indices = [self._chunk_id_to_row_idx[cid] for cid in found_ids]
        idx_array = pa.array(indices, type=pa.int64())
        taken: pa.Table = self._table.take(idx_array)
        return taken.to_pylist(), missing_ids

    def get_neighbors(self, chunk_id: str, window_size: int = 1) -> Dict[str, List[dict]]:
        """Return previous and next neighbors around a chunk_id.

        Traverses the `prev_id` and `next_id` chains up to `window_size` hops.
        The returned lists are ordered nearest-first (distance=1, then 2, ...).

        Args:
            chunk_id: The focal chunk identifier.
            window_size: Number of hops to include in each direction.

        Returns:
            A dictionary with keys 'prev' and 'next', each a list of row dicts.

        Raises:
            ValueError: If the provided chunk_id does not exist.
        """
        if chunk_id not in self._chunk_id_to_row_idx:
            raise ValueError(f"get_neighbors: unknown chunk_id: {chunk_id}")

        prev_ids: List[str] = []
        next_ids: List[str] = []

        # Walk prev chain
        current = chunk_id
        for _ in range(window_size):
            prev = self._chunk_id_to_neighbors[current][0]
            if prev is None:
                break
            prev_ids.append(prev)
            current = prev

        # Walk next chain
        current = chunk_id
        for _ in range(window_size):
            nxt = self._chunk_id_to_neighbors[current][1]
            if nxt is None:
                break
            next_ids.append(nxt)
            current = nxt

        return {
            "prev": self.fetch_chunks_by_ids(prev_ids),
            "next": self.fetch_chunks_by_ids(next_ids),
        }

    def get_chunks_for_section(self, section_path: str) -> List[dict]:
        """Return all chunks belonging to a given section_path.

        Args:
            section_path: Hierarchical section name/path.

        Returns:
            A list of row dicts in document order for that section. Empty if the
            section has no chunks or does not exist.
        """
        chunk_ids = self._section_to_chunk_ids.get(section_path, [])
        if not chunk_ids:
            return []
        return self.fetch_chunks_by_ids(chunk_ids)
