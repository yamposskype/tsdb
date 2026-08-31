"""Metric metadata store.

Keeps track of per-metric type, help text, and unit information.  The store is
purely in-memory; metadata does not need to survive a crash because it is
typically reloaded from config or re-registered at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    UNTYPED = "untyped"


@dataclass
class MetricMetadata:
    """Describes a single metric's type, human-readable help string, and unit."""

    metric: str
    type: MetricType
    help: str = ""
    unit: str = ""


class MetadataStore:
    """Thread-safe(?)-ish in-memory store for :class:`MetricMetadata`.

    We're not adding a lock here because metadata writes are rare (happens at
    startup / config reload) and the GIL gives us enough protection for the
    simple dict ops we do.  If that changes, just add an RLock around the
    mutating methods.
    """

    def __init__(self) -> None:
        self._meta: dict[str, MetricMetadata] = {}

    def set(self, m: MetricMetadata) -> None:
        """Register or overwrite metadata for *m.metric*."""
        self._meta[m.metric] = m

    def get(self, metric: str) -> MetricMetadata | None:
        """Return metadata for *metric*, or ``None`` if not registered."""
        return self._meta.get(metric)

    def list(self) -> list[MetricMetadata]:
        """Return all registered metadata entries, sorted by metric name."""
        return sorted(self._meta.values(), key=lambda m: m.metric)

    def remove(self, metric: str) -> None:
        """Delete the metadata entry for *metric* (no-op if not present)."""
        self._meta.pop(metric, None)
