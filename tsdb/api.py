"""FastAPI HTTP API for tsdb.

All endpoints live under /api/v1/. The storage, ingester, and query engine are
created once at startup via the lifespan context and stored as module-level
singletons — not ideal for testing but fine for a single-process deployment.
"""

import json
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from tsdb.ingest import IngestBatch, IngestPoint, Ingester
from tsdb.query import QueryEngine
from tsdb.storage import Storage
from tsdb.types import AggregationType

# Module-level singletons, initialized in lifespan
_storage: Storage | None = None
_ingester: Ingester | None = None
_engine: QueryEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _storage, _ingester, _engine
    _storage = Storage()
    _ingester = Ingester(_storage)
    _engine = QueryEngine(_storage)
    yield
    # nothing to tear down right now


app = FastAPI(title="tsdb", version="0.1.0", lifespan=lifespan)


def _deps() -> tuple[Storage, Ingester, QueryEngine]:
    if _storage is None or _ingester is None or _engine is None:
        raise HTTPException(status_code=503, detail="storage not initialized yet")
    return _storage, _ingester, _engine


def _parse_labels(raw: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="labels must be valid JSON (e.g. '{\"host\": \"web1\"}')")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="labels must be a JSON object")
    return parsed


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
def health():
    storage, _, _ = _deps()
    return {"status": "ok", "series": storage.series_count()}


@app.get("/api/v1/metrics")
def list_metrics():
    storage, _, _ = _deps()
    keys = storage.list_metrics()
    return {
        "metrics": [
            {"name": k.name, "labels": dict(k.labels)}
            for k in keys
        ]
    }


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.post("/api/v1/ingest", status_code=201)
def ingest(point: IngestPoint):
    _, ingester, _ = _deps()
    try:
        ingester.ingest(point)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/api/v1/ingest/batch", status_code=201)
def ingest_batch(batch: IngestBatch):
    _, ingester, _ = _deps()
    try:
        count = ingester.ingest_batch(batch)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "written": count}


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

@app.get("/api/v1/query")
def instant_query(
    name: str,
    labels: Annotated[str, Query()] = "{}",
    t: Annotated[float | None, Query(description="Unix timestamp, defaults to now")] = None,
):
    """Return the most recent sample value for a metric at a given time."""
    _, _, engine = _deps()
    label_dict = _parse_labels(labels)
    ts = t if t is not None else time.time()

    value = engine.instant_query(name, label_dict, ts)
    if value is None:
        raise HTTPException(status_code=404, detail=f"no data found for '{name}'")
    return {"name": name, "labels": label_dict, "timestamp": ts, "value": value}


@app.get("/api/v1/query_range")
def query_range(
    name: str,
    start: float,
    end: float,
    labels: Annotated[str, Query()] = "{}",
    step: Annotated[float | None, Query(description="Bucket width in seconds for downsampling")] = None,
    agg: Annotated[str, Query(description="Aggregation: mean|sum|min|max|count|last")] = "mean",
):
    _, _, engine = _deps()
    label_dict = _parse_labels(labels)

    try:
        agg_type = AggregationType(agg.lower())
    except ValueError:
        valid = [a.value for a in AggregationType]
        raise HTTPException(status_code=400, detail=f"unknown aggregation '{agg}', valid: {valid}")

    if start >= end:
        raise HTTPException(status_code=400, detail="start must be less than end")

    samples = engine.range_query(name, label_dict, start, end, step=step, agg=agg_type)
    return {
        "name": name,
        "labels": label_dict,
        "samples": [{"timestamp": s.timestamp, "value": s.value} for s in samples],
    }
