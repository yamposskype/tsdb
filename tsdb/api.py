"""FastAPI HTTP API for tsdb.

All endpoints live under /api/v1/. The Engine (WAL + Storage + Checkpoint) is
created once at startup via the lifespan context and stored as a module-level
singleton. The WAL guarantees that every acknowledged write survives a crash.
"""

import json
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from tsdb.alerting.loader import load_rules
from tsdb.alerting.state import AlertState
from tsdb.engine import Engine
from tsdb.index import LabelMatcher, parse_matchers
from tsdb.ingest import IngestBatch, IngestPoint
from tsdb.query import QueryEngine, query_by_matchers
from tsdb.types import AggregationType

# Module-level singletons, initialized in lifespan
_engine: Engine | None = None
_query_engine: QueryEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _query_engine
    _engine = Engine(data_dir="./data")
    _engine.start()
    _query_engine = _engine.query_engine
    yield
    _engine.stop()


app = FastAPI(title="tsdb", version="0.3.0", lifespan=lifespan)


def _get_engine() -> Engine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="storage engine not initialized yet")
    return _engine


def _get_qe() -> QueryEngine:
    if _query_engine is None:
        raise HTTPException(status_code=503, detail="query engine not initialized yet")
    return _query_engine


def _parse_labels(raw: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="labels must be valid JSON (e.g. '{\"host\": \"web1\"}')")
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="labels must be a JSON object")
    return parsed


def _parse_match_param(match_exprs: list[str]) -> list[LabelMatcher]:
    """Parse one or more ``match[]`` query parameter values into LabelMatchers."""
    matchers: list[LabelMatcher] = []
    for expr in match_exprs:
        try:
            matchers.extend(parse_matchers(expr))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid matcher {expr!r}: {exc}") from exc
    return matchers


# ---------------------------------------------------------------------------
# Health / meta
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
def health():
    engine = _get_engine()
    return {"status": "ok", "series": engine.series_count()}


@app.get("/api/v1/metrics")
def list_metrics():
    engine = _get_engine()
    keys = engine.list_metrics()
    return {
        "metrics": [
            {"name": k.name, "labels": dict(k.labels)}
            for k in keys
        ]
    }


# ---------------------------------------------------------------------------
# Label index endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/labels")
def get_label_names():
    """Return all label names present in the index."""
    engine = _get_engine()
    names = engine.storage.all_label_names()
    return {"labels": names}


@app.get("/api/v1/label/{name}/values")
def get_label_values(name: str):
    """Return all values indexed for the given label name."""
    engine = _get_engine()
    values = engine.storage.all_label_values(name)
    return {"label": name, "values": values}


@app.get("/api/v1/series")
def get_series(
    match: Annotated[
        list[str],
        Query(alias="match[]", description="Repeatable label matcher, e.g. match[]={job=\"web\"}"),
    ] = [],
):
    """Return all series matching the given label matchers.

    Each ``match[]`` value is a Prometheus-style label selector such as
    ``{job="web", env=~"prod.*"}``.  Multiple ``match[]`` params are ANDed
    together.  When no matchers are given all indexed series are returned.
    """
    engine = _get_engine()
    matchers = _parse_match_param(match)
    keys = engine.storage.find_series(matchers)
    return {
        "series": [
            {"name": k.name, "labels": dict(k.labels)}
            for k in keys
        ]
    }


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.post("/api/v1/ingest", status_code=201)
def ingest(point: IngestPoint):
    engine = _get_engine()
    ts = point.timestamp if point.timestamp is not None else time.time()
    try:
        engine.write(point.name, point.labels, ts, point.value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/api/v1/ingest/batch", status_code=201)
def ingest_batch(batch: IngestBatch):
    engine = _get_engine()
    now = time.time()
    points = [
        (p.name, p.labels, p.timestamp if p.timestamp is not None else now, p.value)
        for p in batch.points
    ]
    try:
        engine.write_batch(points)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "ok", "written": len(points)}


# ---------------------------------------------------------------------------
# Maintenance: compaction, retention, config
# ---------------------------------------------------------------------------

@app.post("/api/v1/compact", status_code=200)
def compact():
    """Trigger compaction immediately.  Returns the number of chunks merged."""
    engine = _get_engine()
    try:
        merged = engine.compact_now()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", "chunks_merged": merged}


@app.post("/api/v1/retention/apply", status_code=200)
def retention_apply():
    """Apply the retention policy immediately.  Returns eviction stats."""
    engine = _get_engine()
    try:
        stats = engine.apply_retention_now()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", **stats}


@app.get("/api/v1/config", status_code=200)
def get_config():
    """Return current retention policy and compaction configuration."""
    engine = _get_engine()
    policy = engine.retention_manager.policy
    cfg = engine.compactor.config
    return {
        "retention": {
            "name": policy.name,
            "duration_seconds": policy.duration_seconds,
        },
        "compaction": {
            "max_chunk_age_seconds": cfg.max_chunk_age_seconds,
            "target_chunk_samples": cfg.target_chunk_samples,
        },
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

@app.post("/api/v1/checkpoint", status_code=200)
def checkpoint():
    """Manually trigger a checkpoint and WAL rotation."""
    engine = _get_engine()
    try:
        engine.do_checkpoint()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"status": "ok", "message": "checkpoint saved and WAL rotated"}


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
    qe = _get_qe()
    label_dict = _parse_labels(labels)
    ts = t if t is not None else time.time()

    value = qe.instant_query(name, label_dict, ts)
    if value is None:
        raise HTTPException(status_code=404, detail=f"no data found for '{name}'")
    return {"name": name, "labels": label_dict, "timestamp": ts, "value": value}


@app.get("/api/v1/query_range")
def query_range(
    name: Annotated[str | None, Query(description="Metric name (exact). Omit when using match[].")] = None,
    start: float = 0.0,
    end: float = 0.0,
    labels: Annotated[str, Query()] = "{}",
    step: Annotated[float | None, Query(description="Bucket width in seconds for downsampling")] = None,
    agg: Annotated[str, Query(description="Aggregation: mean|sum|min|max|count|last")] = "mean",
    match: Annotated[
        list[str],
        Query(alias="match[]", description="Label matchers, e.g. match[]={job=\"web\"}"),
    ] = [],
):
    """Range query.

    Two modes:
    * Classic — supply ``name`` (and optionally ``labels`` JSON glob dict)
    * Label-index — supply one or more ``match[]`` params instead of ``name``

    When ``match[]`` is supplied it takes precedence over ``name``/``labels``.
    """
    engine = _get_engine()

    try:
        agg_type = AggregationType(agg.lower())
    except ValueError:
        valid = [a.value for a in AggregationType]
        raise HTTPException(status_code=400, detail=f"unknown aggregation '{agg}', valid: {valid}")

    if start >= end:
        raise HTTPException(status_code=400, detail="start must be less than end")

    # --- Label-matcher mode ---
    if match:
        matchers = _parse_match_param(match)
        results = query_by_matchers(engine.storage, matchers, start, end, step=step, agg=agg_type)
        return {
            "results": [
                {
                    "name": qr.key.name,
                    "labels": dict(qr.key.labels),
                    "samples": [{"timestamp": s.timestamp, "value": s.value} for s in qr.samples],
                }
                for qr in results
            ]
        }

    # --- Classic name + label glob mode ---
    if name is None:
        raise HTTPException(status_code=400, detail="supply either 'name' or 'match[]' parameter")

    qe = _get_qe()
    label_dict = _parse_labels(labels)
    samples = qe.range_query(name, label_dict, start, end, step=step, agg=agg_type)
    return {
        "name": name,
        "labels": label_dict,
        "samples": [{"timestamp": s.timestamp, "value": s.value} for s in samples],
    }


# ---------------------------------------------------------------------------
# QL query endpoint
# ---------------------------------------------------------------------------

class _QueryRangeRequest(BaseModel):
    query: str
    start: float = 0.0
    end: float = 0.0
    step: float = 60.0


@app.post("/api/v1/query_range")
def query_range_ql(body: _QueryRangeRequest):  # type: ignore[misc]
    """Evaluate a PromQL-inspired query string over a time range.

    Body JSON::

        {
            "query": "rate(requests_total{job='api'}[5m])",
            "start": 1700000000,
            "end":   1700003600,
            "step":  60
        }

    Returns a list of series, each with ``name``, ``labels``, and ``samples``.
    """
    from tsdb.ql.errors import QueryEvalError, QueryParseError
    from tsdb.ql.evaluator import QueryEvaluator

    engine = _get_engine()

    if body.start >= body.end:
        raise HTTPException(status_code=400, detail="start must be less than end")

    evaluator = QueryEvaluator(engine)
    try:
        results = evaluator.evaluate(body.query, body.start, body.end, body.step)
    except QueryParseError as exc:
        raise HTTPException(status_code=400, detail=f"parse error: {exc}") from exc
    except QueryEvalError as exc:
        raise HTTPException(status_code=400, detail=f"eval error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "results": [
            {
                "name": qr.key.name,
                "labels": dict(qr.key.labels),
                "samples": [
                    {"timestamp": s.timestamp, "value": s.value}
                    for s in qr.samples
                ],
            }
            for qr in results
        ]
    }


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def _alert_to_dict(alert) -> dict:
    return {
        "rule_name": alert.rule_name,
        "labels": alert.labels,
        "state": alert.state.value,
        "value": alert.value,
        "started_at": alert.started_at,
        "fired_at": alert.fired_at,
        "resolved_at": alert.resolved_at,
    }


@app.get("/api/v1/alerts")
def list_alerts():
    """Return all currently active alerts (PENDING, FIRING, or RESOLVED within grace period)."""
    engine = _get_engine()
    alerts = engine.get_alerts()
    return {"alerts": [_alert_to_dict(a) for a in alerts]}


@app.get("/api/v1/alerts/firing")
def list_firing_alerts():
    """Return only FIRING alerts."""
    engine = _get_engine()
    alerts = engine.get_firing_alerts()
    return {"alerts": [_alert_to_dict(a) for a in alerts]}


@app.post("/api/v1/rules", status_code=201)
def create_rules(body: list[dict]):
    """Load one or more alert rules.

    Body is a JSON array of rule objects::

        [
            {
                "name": "high_cpu",
                "expr": "cpu_usage{job='api'}",
                "condition": "gt",
                "threshold": 90,
                "for": "5m",
                "severity": "warning",
                "labels": {"team": "infra"},
                "annotations": {"summary": "CPU above 90%"}
            }
        ]
    """
    engine = _get_engine()
    try:
        rules = load_rules(body)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for rule in rules:
        engine.add_alert_rule(rule)

    return {"status": "ok", "loaded": len(rules)}


@app.get("/api/v1/rules")
def list_rules():
    """Return all registered alert rules."""
    engine = _get_engine()
    rules = engine.list_alert_rules()
    return {
        "rules": [
            {
                "name": r.name,
                "expr": r.expr,
                "condition": r.condition,
                "threshold": r.threshold,
                "for_duration": r.for_duration,
                "severity": r.severity.value,
                "labels": r.labels,
                "annotations": r.annotations,
            }
            for r in rules
        ]
    }


@app.delete("/api/v1/rules/{name}", status_code=200)
def delete_rule(name: str):
    """Remove the alert rule with the given name."""
    engine = _get_engine()
    rules_before = {r.name for r in engine.list_alert_rules()}
    engine.remove_alert_rule(name)
    if name not in rules_before:
        raise HTTPException(status_code=404, detail=f"Rule {name!r} not found")
    return {"status": "ok", "removed": name}


# ---------------------------------------------------------------------------
# Recording rules
# ---------------------------------------------------------------------------

def _recording_group_to_dict(group) -> dict:
    return {
        "name": group.name,
        "interval": group.interval,
        "rules": [
            {
                "record": r.name,
                "expr": r.expr,
                "interval": r.interval,
                "labels": r.labels,
            }
            for r in group.rules
        ],
    }


@app.get("/api/v1/recording_rules")
def list_recording_rules():
    """Return all loaded recording rule groups and their rules."""
    engine = _get_engine()
    groups = engine.list_recording_groups()
    return {"groups": [_recording_group_to_dict(g) for g in groups]}


@app.post("/api/v1/recording_rules", status_code=201)
def load_recording_rules(body: dict):
    """Load a recording rule group config.

    Body JSON follows the Prometheus rule file format::

        {
            "groups": [
                {
                    "name": "aggregations",
                    "interval": "1m",
                    "rules": [
                        {
                            "record": "job:requests:rate5m",
                            "expr": "rate(requests_total{job='api'}[5m])",
                            "labels": {"env": "prod"}
                        }
                    ]
                }
            ]
        }
    """
    engine = _get_engine()
    try:
        engine.load_recording_rules(body)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    groups = engine.list_recording_groups()
    return {
        "status": "ok",
        "groups_loaded": len(groups),
        "rules_loaded": sum(len(g.rules) for g in groups),
    }


@app.post("/api/v1/recording_rules/{group_name}/evaluate", status_code=200)
def evaluate_recording_group(group_name: str):
    """Force-evaluate a recording rule group immediately.

    Useful for testing or for backfilling a newly added group without
    waiting for its next scheduled evaluation.
    """
    engine = _get_engine()
    try:
        found = engine.evaluate_recording_group_now(group_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Recording rule group {group_name!r} not found",
        )
    return {"status": "ok", "group": group_name, "evaluated": True}
