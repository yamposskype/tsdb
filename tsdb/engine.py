"""Unified storage engine combining WAL, in-memory Storage, and Checkpoint.

Write path:
  1. Write to WAL (durable on disk before acknowledgement)
  2. Write to in-memory Storage

On startup:
  1. Load checkpoint (if one exists) into Storage
  2. Replay WAL entries that came after the checkpoint

The WAL is automatically rotated after a checkpoint so it doesn't grow
without bound. A checkpoint is triggered automatically once the WAL
exceeds WAL_CHECKPOINT_THRESHOLD bytes.
"""

import logging
import time
from pathlib import Path

from tsdb.background import BackgroundWorker
from tsdb.checkpoint import Checkpoint
from tsdb.compaction import CompactionConfig, Compactor
from tsdb.query import QueryEngine
from tsdb.retention import DEFAULT_RETENTION, RetentionManager, RetentionPolicy
from tsdb.storage import Storage
from tsdb.types import MetricKey, QueryResult, Sample
from tsdb.wal import WAL, WALEntry

WAL_CHECKPOINT_THRESHOLD = 10 * 1024 * 1024  # 10 MB

logger = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        data_dir: str | Path,
        retention_policy: RetentionPolicy | None = None,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._storage = Storage()
        self._wal = WAL(self._data_dir)
        self._checkpoint = Checkpoint(self._data_dir, self._storage)
        self._query_engine = QueryEngine(self._storage)

        policy = retention_policy if retention_policy is not None else DEFAULT_RETENTION
        config = compaction_config if compaction_config is not None else CompactionConfig()
        self._retention_manager = RetentionManager(self._storage, policy)
        self._compactor = Compactor(self._storage, config)
        self._worker = BackgroundWorker()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Restore state from disk, then open the WAL for new writes."""
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: load last checkpoint into memory
        loaded = self._checkpoint.load()
        if loaded:
            count = self._storage.series_count()
            logger.info("Checkpoint loaded: %d series restored", count)
        else:
            logger.info("No checkpoint found; starting from empty storage")

        # Step 2: replay WAL on top of checkpoint
        entries = self._wal.replay()
        recovered = 0
        for entry in entries:
            if entry.op == "write":
                self._storage.write(entry.name, entry.labels, entry.timestamp, entry.value)
                recovered += 1

        if recovered:
            logger.info("WAL replay: %d entries recovered", recovered)
        else:
            logger.info("WAL replay: nothing to recover")

        # Step 3: open WAL for new writes
        self._wal.open()

        # Step 4: start background maintenance tasks
        self._worker.register("retention", 3600, self._retention_manager.apply)
        self._worker.register("compaction", 1800, self._compactor.compact_all)
        self._worker.register("checkpoint", 60, self.checkpoint_if_needed)
        self._worker.start()

    def stop(self) -> None:
        """Checkpoint current state and close the WAL cleanly."""
        self._worker.stop()
        self._checkpoint.save()
        logger.info("Checkpoint saved on shutdown")
        self._wal.close()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write(self, name: str, labels: dict, timestamp: float, value: float) -> None:
        """WAL-before-data write: log first, then apply to in-memory store."""
        self._wal.write(name, labels, timestamp, value)
        self._storage.write(name, labels, timestamp, value)
        self.checkpoint_if_needed()

    def write_batch(self, points: list) -> None:
        """Batch write: single WAL flush, single storage lock acquisition.

        Each item in `points` must be a (name, labels, timestamp, value) tuple
        or an object with those attributes.
        """
        wal_entries: list[WALEntry] = []
        storage_rows: list[tuple] = []

        for point in points:
            if isinstance(point, (list, tuple)):
                name, labels, timestamp, value = point
            else:
                name = point.name
                labels = point.labels
                timestamp = point.timestamp if point.timestamp is not None else time.time()
                value = point.value
            wal_entries.append(WALEntry(op="write", name=name, labels=labels, timestamp=timestamp, value=value))
            storage_rows.append((name, labels, timestamp, value))

        self._wal.write_batch(wal_entries)
        self._storage.write_batch(storage_rows)
        self.checkpoint_if_needed()

    # ------------------------------------------------------------------
    # Query path
    # ------------------------------------------------------------------

    def query(self, name: str, labels: dict, start: float, end: float) -> list[Sample]:
        return self._storage.query(name, labels, start, end)

    def query_range(
        self,
        name: str,
        label_matchers: dict[str, str],
        start: float,
        end: float,
    ) -> list[QueryResult]:
        return self._storage.query_range(name, label_matchers, start, end)

    def list_metrics(self) -> list[MetricKey]:
        return self._storage.list_metrics()

    def series_count(self) -> int:
        return self._storage.series_count()

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def checkpoint_if_needed(self) -> None:
        """Save a checkpoint and rotate the WAL when the WAL exceeds the threshold."""
        if self._wal.size_bytes() >= WAL_CHECKPOINT_THRESHOLD:
            self.do_checkpoint()

    def do_checkpoint(self) -> None:
        """Force a checkpoint and rotate the WAL now."""
        self._checkpoint.save()
        self._wal.rotate()
        logger.info("Checkpoint saved and WAL rotated (WAL size exceeded threshold)")

    # ------------------------------------------------------------------
    # Manual maintenance triggers
    # ------------------------------------------------------------------

    def compact_now(self) -> int:
        """Run compaction immediately and return the number of chunks merged."""
        return self._compactor.compact_all()

    def apply_retention_now(self) -> dict:
        """Apply the retention policy immediately and return eviction stats."""
        self._retention_manager.apply()
        return {
            "samples_removed": self._retention_manager.samples_removed(),
            "bytes_freed": self._retention_manager.bytes_freed_estimate(),
            "series_removed": self._retention_manager.series_removed(),
        }

    # ------------------------------------------------------------------
    # Expose underlying components for convenience
    # ------------------------------------------------------------------

    @property
    def query_engine(self) -> QueryEngine:
        return self._query_engine

    @property
    def storage(self) -> Storage:
        return self._storage

    @property
    def retention_manager(self) -> RetentionManager:
        return self._retention_manager

    @property
    def compactor(self) -> Compactor:
        return self._compactor
