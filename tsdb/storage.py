"""Main storage engine.

Thread-safe via an RLock so multiple ingest threads don't stomp on each other.
query_range supports fnmatch-style glob patterns in label matchers, so you can
do things like {"host": "web-*"} to match all web hosts.
"""

import fnmatch
import threading

from tsdb.chunk import ChunkStore
from tsdb.types import MetricKey, QueryResult, Sample, make_key


class Storage:
    def __init__(self) -> None:
        self._series: dict[MetricKey, ChunkStore] = {}
        self._lock = threading.RLock()

    def write(self, name: str, labels: dict[str, str], timestamp: float, value: float) -> None:
        key = make_key(name, labels)
        with self._lock:
            store = self._get_or_create(key)
            store.append(Sample(timestamp=timestamp, value=value))

    def write_batch(self, samples: list[tuple[str, dict, float, float]]) -> None:
        """Bulk write — acquires the lock once for the whole batch."""
        with self._lock:
            for name, labels, timestamp, value in samples:
                key = make_key(name, labels)
                store = self._get_or_create(key)
                store.append(Sample(timestamp=timestamp, value=value))

    def query(self, name: str, labels: dict[str, str], start: float, end: float) -> list[Sample]:
        key = make_key(name, labels)
        with self._lock:
            store = self._series.get(key)
            if store is None:
                return []
            return store.query(start, end)

    def query_range(
        self,
        name: str,
        label_matchers: dict[str, str],
        start: float,
        end: float,
    ) -> list[QueryResult]:
        """Query all series matching name + label glob patterns."""
        results: list[QueryResult] = []
        with self._lock:
            for key, store in self._series.items():
                if key.name != name:
                    continue
                key_labels = dict(key.labels)
                if not _labels_match(key_labels, label_matchers):
                    continue
                samples = store.query(start, end)
                results.append(QueryResult(key=key, samples=samples))
        return results

    def list_metrics(self) -> list[MetricKey]:
        with self._lock:
            return list(self._series.keys())

    def series_count(self) -> int:
        with self._lock:
            return len(self._series)

    def _get_or_create(self, key: MetricKey) -> ChunkStore:
        if key not in self._series:
            self._series[key] = ChunkStore()
        return self._series[key]


def _labels_match(key_labels: dict[str, str], matchers: dict[str, str]) -> bool:
    """Return True if all matcher patterns match the corresponding key label values."""
    for label_name, pattern in matchers.items():
        value = key_labels.get(label_name, "")
        if not fnmatch.fnmatch(value, pattern):
            return False
    return True
