"""Write-Ahead Log (WAL) for crash durability.

Every write is appended to the WAL file before being applied to the in-memory
store. On restart, the WAL is replayed to reconstruct writes that were not yet
captured in a checkpoint. Each line is a complete JSON record so partial writes
at the tail (e.g. due to a crash mid-line) are silently skipped during replay.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

WAL_FILENAME = "wal.log"
WAL_OLD_FILENAME = "wal.old"


@dataclass
class WALEntry:
    op: str                   # always "write" for now
    name: str
    labels: dict[str, str]
    timestamp: float
    value: float


class WAL:
    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / WAL_FILENAME
        self._file: IO | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        """Open (or create) the WAL file in append mode."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")  # noqa: WPS515

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None

    def write(self, name: str, labels: dict, timestamp: float, value: float) -> None:
        """Append a single WAL entry and flush to disk."""
        entry = WALEntry(op="write", name=name, labels=labels, timestamp=timestamp, value=value)
        line = json.dumps(asdict(entry), separators=(",", ":")) + "\n"
        with self._lock:
            if self._file is None:
                raise RuntimeError("WAL is not open; call open() first")
            self._file.write(line)
            self._file.flush()
            os.fsync(self._file.fileno())

    def write_batch(self, entries: list[WALEntry]) -> None:
        """Append multiple entries with a single flush."""
        if not entries:
            return
        lines = "".join(
            json.dumps(asdict(e), separators=(",", ":")) + "\n" for e in entries
        )
        with self._lock:
            if self._file is None:
                raise RuntimeError("WAL is not open; call open() first")
            self._file.write(lines)
            self._file.flush()
            os.fsync(self._file.fileno())

    def replay(self) -> list[WALEntry]:
        """Read all valid entries from the WAL file from the beginning.

        Corrupt or truncated lines at the tail are silently skipped so a crash
        mid-write does not prevent startup.
        """
        if not self.path.exists():
            return []
        entries: list[WALEntry] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                    entries.append(
                        WALEntry(
                            op=data["op"],
                            name=data["name"],
                            labels=data["labels"],
                            timestamp=float(data["timestamp"]),
                            value=float(data["value"]),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # Skip corrupt lines — likely a partial write from a crash
                    continue
        return entries

    def rotate(self) -> None:
        """Archive the current WAL and start a fresh one.

        Called after a successful checkpoint so the log doesn't grow unboundedly.
        """
        with self._lock:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None

            old_path = self.path.parent / WAL_OLD_FILENAME
            if old_path.exists():
                old_path.unlink()
            if self.path.exists():
                self.path.rename(old_path)

            self._file = open(self.path, "a", encoding="utf-8")

    def size_bytes(self) -> int:
        """Return the current WAL file size in bytes, or 0 if the file does not exist."""
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0
