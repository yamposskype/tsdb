from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


@dataclass
class Sample:
    """A single timestamped data point. timestamp is Unix seconds (float for sub-second)."""
    timestamp: float
    value: float


@dataclass
class TimeSeries:
    """Named metric with labels and its list of samples."""
    name: str
    labels: dict[str, str]
    samples: list[Sample] = field(default_factory=list)


class MetricKey(NamedTuple):
    """Hashable identity of a metric series — name + frozen label set."""
    name: str
    labels: frozenset[tuple[str, str]]


def make_key(name: str, labels: dict[str, str]) -> MetricKey:
    return MetricKey(name=name, labels=frozenset(labels.items()))


@dataclass
class QueryResult:
    """Result of a range query for one metric key."""
    key: MetricKey
    samples: list[Sample]


class AggregationType(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    LAST = "last"
