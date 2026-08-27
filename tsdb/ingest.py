"""Ingestion layer: validates incoming data and writes it to storage."""

import time

from pydantic import BaseModel, field_validator

from tsdb.storage import Storage


class IngestPoint(BaseModel):
    name: str
    labels: dict[str, str] = {}
    timestamp: float | None = None  # None means "use server time"
    value: float

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("metric name cannot be empty")
        return v


class IngestBatch(BaseModel):
    points: list[IngestPoint]


class Ingester:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def ingest(self, point: IngestPoint) -> None:
        ts = point.timestamp if point.timestamp is not None else time.time()
        self.storage.write(point.name, point.labels, ts, point.value)

    def ingest_batch(self, batch: IngestBatch) -> int:
        """Write all points in one storage call. Returns count of points written."""
        now = time.time()
        raw = [
            (p.name, p.labels, p.timestamp if p.timestamp is not None else now, p.value)
            for p in batch.points
        ]
        self.storage.write_batch(raw)
        return len(raw)
