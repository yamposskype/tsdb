"""Retention policy enforcement.

Evicts data older than the configured duration from in-memory storage.
Accesses Storage internals directly since both modules live in the same
package and adding a proliferation of public eviction hooks would not
justify the extra API surface.
"""

import logging
import time
from dataclasses import dataclass

from tsdb.chunk import Chunk
from tsdb.storage import Storage
from tsdb.types import MetricKey, Sample

logger = logging.getLogger(__name__)

# float timestamp (8 bytes) + float value (8 bytes)
_BYTES_PER_SAMPLE = 16


@dataclass
class RetentionPolicy:
    duration_seconds: float
    name: str = "default"


DEFAULT_RETENTION = RetentionPolicy(duration_seconds=7 * 24 * 3600, name="default")


def _build_chunk(samples: list[Sample]) -> Chunk:
    """Create a Chunk whose samples are set directly (bypasses the size cap)."""
    chunk = Chunk()
    if not samples:
        return chunk
    chunk.samples = list(samples)
    chunk.min_ts = min(s.timestamp for s in samples)
    chunk.max_ts = max(s.timestamp for s in samples)
    return chunk


class RetentionManager:
    def __init__(self, storage: Storage, policy: RetentionPolicy) -> None:
        self._storage = storage
        self._policy = policy
        self._samples_removed: int = 0
        self._series_removed: int = 0

    @property
    def policy(self) -> RetentionPolicy:
        return self._policy

    def apply(self) -> None:
        """Remove all samples older than now - policy.duration_seconds."""
        self._samples_removed = 0
        self._series_removed = 0

        cutoff = time.time() - self._policy.duration_seconds
        # Snapshot the key list outside the per-key lock to avoid holding the
        # global lock during chunk rebuilds.
        keys = self._storage.list_metrics()
        for key in keys:
            self._evict_old_chunks(key, cutoff)

        if self._samples_removed or self._series_removed:
            logger.info(
                "Retention (%s): evicted %d samples, freed ~%d bytes, "
                "removed %d empty series (cutoff=%.0f)",
                self._policy.name,
                self._samples_removed,
                self.bytes_freed_estimate(),
                self._series_removed,
                cutoff,
            )

    def _evict_old_chunks(self, key: MetricKey, cutoff: float) -> None:
        storage = self._storage
        with storage._lock:
            store = storage._series.get(key)
            if store is None:
                return

            new_chunks: list[Chunk] = []
            removed = 0

            for chunk in store.chunks:
                if chunk.size() == 0:
                    continue

                if chunk.max_ts < cutoff:
                    # Entire chunk predates the cutoff — drop it.
                    removed += chunk.size()
                    continue

                if chunk.min_ts < cutoff:
                    # Partially overlapping — keep only samples at or after cutoff.
                    fresh = [s for s in chunk.samples if s.timestamp >= cutoff]
                    dropped = len(chunk.samples) - len(fresh)
                    removed += dropped
                    if fresh:
                        new_chunks.append(_build_chunk(fresh))
                    # If fresh is empty the whole chunk was before cutoff; skip it.
                else:
                    new_chunks.append(chunk)

            self._samples_removed += removed

            if not new_chunks:
                # All data for this series has expired.
                del storage._series[key]
                storage._index.remove(key)
                self._series_removed += 1
                return

            store.chunks = new_chunks
            store.head = new_chunks[-1]

    def bytes_freed_estimate(self) -> int:
        """Rough byte count freed: samples_removed * 16 bytes each."""
        return self._samples_removed * _BYTES_PER_SAMPLE

    def series_removed(self) -> int:
        """Number of series that became completely empty and were deleted."""
        return self._series_removed

    def samples_removed(self) -> int:
        """Number of individual samples evicted in the last apply() call."""
        return self._samples_removed
