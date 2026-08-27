"""Chunk-based append-only storage for time-series samples.

Each Chunk holds up to CHUNK_SIZE samples. When a chunk fills up, a new one is
created. ChunkStore manages the chain of chunks for a single metric series and
skips chunks that don't overlap the query time range, which keeps range reads fast.
"""

import bisect

from tsdb.types import Sample

CHUNK_SIZE = 120


class Chunk:
    def __init__(self) -> None:
        self.samples: list[Sample] = []
        self.min_ts: float = float("inf")
        self.max_ts: float = float("-inf")

    def append(self, sample: Sample) -> bool:
        """Try to append a sample. Returns False if the chunk is full."""
        if self.is_full():
            return False
        self.samples.append(sample)
        if sample.timestamp < self.min_ts:
            self.min_ts = sample.timestamp
        if sample.timestamp > self.max_ts:
            self.max_ts = sample.timestamp
        return True

    def query(self, start: float, end: float) -> list[Sample]:
        """Binary search for samples in [start, end]."""
        timestamps = [s.timestamp for s in self.samples]
        lo = bisect.bisect_left(timestamps, start)
        hi = bisect.bisect_right(timestamps, end)
        return self.samples[lo:hi]

    def is_full(self) -> bool:
        return len(self.samples) >= CHUNK_SIZE

    def size(self) -> int:
        return len(self.samples)


class ChunkStore:
    """Manages an ordered list of Chunks for a single metric series."""

    def __init__(self) -> None:
        self.head = Chunk()
        self.chunks: list[Chunk] = [self.head]

    def append(self, sample: Sample) -> None:
        if not self.head.append(sample):
            self.head = Chunk()
            self.chunks.append(self.head)
            self.head.append(sample)

    def query(self, start: float, end: float) -> list[Sample]:
        result: list[Sample] = []
        for chunk in self.chunks:
            # Skip chunks that don't overlap the query window
            if chunk.size() == 0:
                continue
            if chunk.max_ts < start or chunk.min_ts > end:
                continue
            result.extend(chunk.query(start, end))
        return result

    def total_samples(self) -> int:
        return sum(c.size() for c in self.chunks)
