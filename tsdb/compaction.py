"""Chunk compaction.

Merges adjacent small chunks into larger ones to reduce per-chunk overhead
and improve query scan efficiency. Compaction never drops data — it only
reorganises how samples are grouped.
"""

import logging
from dataclasses import dataclass, field

from tsdb.chunk import Chunk
from tsdb.storage import Storage
from tsdb.types import MetricKey, Sample

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    max_chunk_age_seconds: float = 3600
    target_chunk_samples: int = 240


def _build_chunk(samples: list[Sample]) -> Chunk:
    """Create a Chunk directly from a sample list, bypassing the size cap."""
    chunk = Chunk()
    if not samples:
        return chunk
    chunk.samples = list(samples)
    chunk.min_ts = min(s.timestamp for s in samples)
    chunk.max_ts = max(s.timestamp for s in samples)
    return chunk


class Compactor:
    def __init__(self, storage: Storage, config: CompactionConfig) -> None:
        self._storage = storage
        self._config = config

    @property
    def config(self) -> CompactionConfig:
        return self._config

    def compact_series(self, key: MetricKey) -> int:
        """Merge adjacent small chunks for one series.

        Two consecutive chunks are merged when their combined sample count is
        below target_chunk_samples.  The merge is applied repeatedly (one pass
        left-to-right) until no further merges are possible.

        Returns the number of chunks merged away (i.e. how many fewer chunks
        exist after compaction compared to before).
        """
        storage = self._storage
        with storage._lock:
            store = storage._series.get(key)
            if store is None:
                return 0

            original_count = len(store.chunks)
            target = self._config.target_chunk_samples

            merged = self._merge_pass(store.chunks, target)
            if len(merged) == original_count:
                return 0

            store.chunks = merged
            store.head = merged[-1]
            return original_count - len(merged)

    def _merge_pass(self, chunks: list[Chunk], target: int) -> list[Chunk]:
        """Single left-to-right pass merging adjacent small chunks."""
        if not chunks:
            return chunks

        result: list[Chunk] = [chunks[0]]
        for chunk in chunks[1:]:
            prev = result[-1]
            if prev.size() + chunk.size() < target:
                # Merge chunk into prev.
                result[-1] = self._merge_chunks(prev, chunk)
            else:
                result.append(chunk)
        return result

    def compact_all(self) -> int:
        """Run compact_series on every series.  Returns total chunks merged."""
        keys = self._storage.list_metrics()
        total_merged = 0
        for key in keys:
            total_merged += self.compact_series(key)
        if total_merged:
            logger.info("Compaction: merged %d chunks across %d series", total_merged, len(keys))
        return total_merged

    def _merge_chunks(self, c1: Chunk, c2: Chunk) -> Chunk:
        """Create a new Chunk containing all samples from c1 and c2, sorted by timestamp."""
        combined = sorted(c1.samples + c2.samples, key=lambda s: s.timestamp)
        return _build_chunk(combined)
