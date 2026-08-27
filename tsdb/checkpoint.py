"""Checkpoint mechanism for point-in-time snapshots of in-memory storage.

A checkpoint is a JSON file that captures the complete state of all series so
that on restart the WAL only needs to replay entries written after the last
checkpoint. Writes are atomic: the new data goes to a temp file first, then
an os.replace() swaps it into place so readers never see a partial file.
"""

import json
import os
from pathlib import Path

from tsdb.storage import Storage
from tsdb.types import MetricKey, Sample, make_key

CHECKPOINT_FILENAME = "checkpoint.json"


class Checkpoint:
    def __init__(self, data_dir: str | Path, storage: Storage) -> None:
        self._path = Path(data_dir) / CHECKPOINT_FILENAME
        self._tmp_path = Path(data_dir) / (CHECKPOINT_FILENAME + ".tmp")
        self._storage = storage

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Snapshot current storage to disk atomically."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        metrics = self._storage.list_metrics()
        serialized = []
        for key in metrics:
            samples = self._storage.query(
                key.name,
                dict(key.labels),
                float("-inf"),
                float("inf"),
            )
            serialized.append(
                {
                    "key": self._serialize_key(key),
                    "samples": [{"timestamp": s.timestamp, "value": s.value} for s in samples],
                }
            )

        data = {"series": serialized}
        with open(self._tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())

        os.replace(self._tmp_path, self._path)

    def load(self) -> bool:
        """Restore storage state from the checkpoint file.

        Returns True if a checkpoint was found and loaded, False if none exists.
        """
        if not self._path.exists():
            return False

        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return False

        for entry in data.get("series", []):
            try:
                key = self._deserialize_key(entry["key"])
            except (KeyError, TypeError, ValueError):
                continue
            for raw_sample in entry.get("samples", []):
                try:
                    self._storage.write(
                        key.name,
                        dict(key.labels),
                        float(raw_sample["timestamp"]),
                        float(raw_sample["value"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue

        return True

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _serialize_key(self, key: MetricKey) -> dict:
        return {"name": key.name, "labels": dict(key.labels)}

    def _deserialize_key(self, d: dict) -> MetricKey:
        return make_key(d["name"], d["labels"])
