"""Query engine: aggregations, downsampling, rate computation, and query helpers."""

import math
from typing import TYPE_CHECKING

from tsdb.types import AggregationType, QueryResult, Sample

if TYPE_CHECKING:
    from tsdb.index import LabelMatcher
    from tsdb.storage import Storage


def aggregate(samples: list[Sample], agg: AggregationType) -> float:
    if not samples:
        return float("nan")
    values = [s.value for s in samples]
    match agg:
        case AggregationType.MEAN:
            return sum(values) / len(values)
        case AggregationType.SUM:
            return sum(values)
        case AggregationType.MIN:
            return min(values)
        case AggregationType.MAX:
            return max(values)
        case AggregationType.COUNT:
            return float(len(values))
        case AggregationType.LAST:
            return values[-1]
        case _:
            raise ValueError(f"Unknown aggregation: {agg}")


def downsample(samples: list[Sample], step: float, agg: AggregationType) -> list[Sample]:
    """Group samples into fixed-width time buckets and aggregate each bucket.

    Buckets start at the timestamp of the first sample, aligned to `step`.
    Empty buckets are omitted from the output.
    """
    if not samples or step <= 0:
        return []

    origin = math.floor(samples[0].timestamp / step) * step
    result: list[Sample] = []

    # Walk samples into buckets
    current_bucket_start = origin
    bucket: list[Sample] = []

    for s in samples:
        # Advance bucket until this sample fits
        while s.timestamp >= current_bucket_start + step:
            if bucket:
                result.append(Sample(timestamp=current_bucket_start, value=aggregate(bucket, agg)))
            bucket = []
            current_bucket_start += step
        bucket.append(s)

    if bucket:
        result.append(Sample(timestamp=current_bucket_start, value=aggregate(bucket, agg)))

    return result


def rate(samples: list[Sample]) -> list[Sample]:
    """Compute per-second rate of change for a counter metric.

    Handles counter resets by treating a decrease as a reset to zero and
    measuring the new value from 0.
    """
    if len(samples) < 2:
        return []

    result: list[Sample] = []
    for i in range(1, len(samples)):
        dt = samples[i].timestamp - samples[i - 1].timestamp
        if dt <= 0:
            continue  # skip out-of-order or duplicate timestamps
        delta = samples[i].value - samples[i - 1].value
        if delta < 0:
            # counter reset — measure from 0
            delta = samples[i].value
        result.append(Sample(timestamp=samples[i].timestamp, value=delta / dt))

    return result


def query_by_matchers(
    storage: "Storage",
    matchers: "list[LabelMatcher]",
    start: float,
    end: float,
    step: float | None = None,
    agg: AggregationType = AggregationType.MEAN,
) -> list[QueryResult]:
    """Find series matching *matchers* and query each over [start, end].

    If *step* is given the samples for every matched series are downsampled
    into fixed-width time buckets of width *step* using *agg*.
    """
    results = storage.query_by_matchers(matchers, start, end)
    if step is not None:
        downsampled: list[QueryResult] = []
        for qr in results:
            ds_samples = downsample(qr.samples, step, agg) if qr.samples else []
            downsampled.append(QueryResult(key=qr.key, samples=ds_samples))
        return downsampled
    return results


class QueryEngine:
    def __init__(self, storage: "Storage") -> None:
        self.storage = storage

    def range_query(
        self,
        name: str,
        labels: dict[str, str],
        start: float,
        end: float,
        step: float | None = None,
        agg: AggregationType = AggregationType.MEAN,
    ) -> list[Sample]:
        samples = self.storage.query(name, labels, start, end)
        if step is not None and samples:
            samples = downsample(samples, step, agg)
        return samples

    def instant_query(self, name: str, labels: dict[str, str], timestamp: float) -> float | None:
        """Return the most recent sample value at or before `timestamp`.

        Looks back up to 5 minutes. Returns None if no data found.
        """
        lookback = 300.0
        samples = self.storage.query(name, labels, timestamp - lookback, timestamp)
        if not samples:
            return None
        # Samples from storage are in insertion order; find the latest one <= timestamp
        relevant = [s for s in samples if s.timestamp <= timestamp]
        if not relevant:
            return None
        return max(relevant, key=lambda s: s.timestamp).value
