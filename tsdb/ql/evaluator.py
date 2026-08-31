"""Query evaluator: walks the AST and retrieves / transforms data from the engine.

Evaluation model
----------------
* A *selector* (MetricSelector) fetches all raw samples in [start, end] for
  the matching series.
* A *range query* (RangeQuery) is not evaluated on its own — it is consumed by
  window functions (rate, avg_over_time, …) that slide a window over the raw
  data at each output step.
* Aggregation functions (sum, avg, min, max, count) collapse the series
  dimension: they take the instant value of each matching series at every step
  and combine them.
* Binary operations work element-wise: series vs scalar, or two series matched
  by their label fingerprint.

Internal return type: ``list[QueryResult]`` for vector results, ``float`` for
scalar results.  The public :meth:`QueryEvaluator.evaluate` always returns
``list[QueryResult]``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Union

from tsdb.ql.ast import (
    BinaryOp,
    FunctionCall,
    LabelMatcher,
    MetricSelector,
    NumberLiteral,
    RangeQuery,
)
from tsdb.ql.errors import QueryEvalError
from tsdb.ql.parser import parse_query
from tsdb.types import MetricKey, QueryResult, Sample

if TYPE_CHECKING:
    from tsdb.engine import Engine

# Internal result type: a vector of series or a scalar number.
_EvalResult = Union[list[QueryResult], float]

# Default look-back window used when evaluating instant selectors (seconds).
_INSTANT_LOOKBACK = 300.0


class QueryEvaluator:
    """Evaluates query strings against a running :class:`~tsdb.engine.Engine`.

    Usage::

        evaluator = QueryEvaluator(engine)
        results   = evaluator.evaluate("rate(requests[5m])", start, end, step=60)
    """

    def __init__(self, engine: "Engine") -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query_str: str,
        start: float,
        end: float,
        step: float = 60.0,
    ) -> list[QueryResult]:
        """Parse *query_str* and evaluate it over the time range [start, end].

        *step* is the interval between output timestamps in seconds.

        Returns a list of :class:`~tsdb.types.QueryResult`, one per output series.
        Raises :class:`~tsdb.ql.errors.QueryParseError` or
        :class:`~tsdb.ql.errors.QueryEvalError` on failure.
        """
        ast = parse_query(query_str)
        result = self._eval(ast, start, end, step)

        if isinstance(result, float):
            # Wrap a scalar in a synthetic single-series result.
            key = MetricKey(name="scalar", labels=frozenset())
            steps = _make_steps(start, end, step)
            samples = [Sample(timestamp=t, value=result) for t in steps]
            return [QueryResult(key=key, samples=samples)]

        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _eval(self, node, start: float, end: float, step: float) -> _EvalResult:
        if isinstance(node, MetricSelector):
            return self._eval_selector(node, start, end, step)
        if isinstance(node, RangeQuery):
            return self._eval_range_query(node, start, end, step)
        if isinstance(node, FunctionCall):
            return self._eval_function(node, start, end, step)
        if isinstance(node, BinaryOp):
            return self._eval_binary_op(node, start, end, step)
        if isinstance(node, NumberLiteral):
            return node.value
        raise QueryEvalError(f"Unknown AST node type: {type(node).__name__}")

    # ------------------------------------------------------------------
    # Selector
    # ------------------------------------------------------------------

    def _eval_selector(
        self, node: MetricSelector, start: float, end: float, step: float
    ) -> list[QueryResult]:
        """Fetch all raw samples in [start, end] for the matching series."""
        matchers = _build_index_matchers(node)
        return self._engine.storage.query_by_matchers(matchers, start, end)

    def _eval_range_query(
        self, node: RangeQuery, start: float, end: float, step: float
    ) -> list[QueryResult]:
        """Fetch raw data for the extended range needed by window functions.

        This is called when a RangeQuery appears outside a window function
        (unusual but legal).  Window functions call this themselves with the
        appropriate extended range.
        """
        fetch_start = start - node.range_duration - node.offset
        fetch_end = end - node.offset
        matchers = _build_index_matchers(node.selector)
        return self._engine.storage.query_by_matchers(matchers, fetch_start, fetch_end)

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def _eval_function(
        self, node: FunctionCall, start: float, end: float, step: float
    ) -> _EvalResult:
        fn = node.function
        steps = _make_steps(start, end, step)

        # Window / rate functions — require a RangeQuery argument.
        window_fns = frozenset({
            "rate", "irate", "delta", "increase",
            "changes", "resets",
            "avg_over_time", "min_over_time", "max_over_time", "sum_over_time",
        })
        if fn in window_fns:
            return self._eval_window_function(fn, node, start, steps)

        # Aggregation functions — collapse the series dimension.
        agg_fns = frozenset({"sum", "avg", "min", "max", "count"})
        if fn in agg_fns:
            if not node.args:
                raise QueryEvalError(f"Function '{fn}' requires at least one argument")
            inner = self._eval(node.args[0], start, end, step)
            if not isinstance(inner, list):
                raise QueryEvalError(f"Function '{fn}' expects a series, got a scalar")
            return self._aggregate_series(fn, inner, steps, node.by, node.without)

        # Histogram helpers.
        if fn == "histogram_quantile":
            return self._eval_histogram_quantile(node, start, end, step, steps)

        if fn in ("histogram_count", "histogram_sum"):
            if not node.args:
                raise QueryEvalError(f"Function '{fn}' requires a series argument")
            inner = self._eval(node.args[0], start, end, step)
            if not isinstance(inner, list):
                raise QueryEvalError(f"Function '{fn}' expects a series, got a scalar")
            return self._histogram_simple(fn, inner, steps)

        raise QueryEvalError(f"Unknown function: {fn!r}")

    # --- Window functions (rate, avg_over_time, etc.) ------------------

    def _eval_window_function(
        self,
        fn: str,
        node: FunctionCall,
        start: float,
        steps: list[float],
    ) -> list[QueryResult]:
        if not node.args:
            raise QueryEvalError(f"Function '{fn}' requires a range-selector argument")
        arg = node.args[0]
        if not isinstance(arg, RangeQuery):
            raise QueryEvalError(
                f"Function '{fn}' requires a range-selector like metric[5m], "
                f"got {type(arg).__name__}"
            )

        duration = arg.range_duration
        offset = arg.offset
        end = steps[-1] if steps else start

        # Fetch enough raw data to cover every window at every step.
        fetch_start = steps[0] - duration - offset if steps else start - duration - offset
        fetch_end = end - offset
        matchers = _build_index_matchers(arg.selector)
        raw_results = self._engine.storage.query_by_matchers(
            matchers, fetch_start, fetch_end
        )

        output: list[QueryResult] = []
        for qr in raw_results:
            all_samples = sorted(qr.samples, key=lambda s: s.timestamp)
            result_samples: list[Sample] = []

            for t in steps:
                window_start = t - duration - offset
                window_end = t - offset
                window = [
                    s for s in all_samples
                    if window_start <= s.timestamp <= window_end
                ]
                val = _apply_window_fn(fn, window, duration)
                if val is not None:
                    result_samples.append(Sample(timestamp=t, value=val))

            output.append(QueryResult(key=qr.key, samples=result_samples))

        return output

    # --- Aggregation functions (sum, avg, min, max, count) -------------

    def _aggregate_series(
        self,
        fn: str,
        series_list: list[QueryResult],
        steps: list[float],
        by_labels: list[str] | None,
        without_labels: list[str] | None,
    ) -> list[QueryResult]:
        """Aggregate across series at each step, grouping by label dimensions."""
        # Group series by their aggregation key.
        groups: dict[frozenset, list[QueryResult]] = {}
        for qr in series_list:
            labels = dict(qr.key.labels)
            if by_labels is not None:
                group_key = frozenset(
                    (k, v) for k, v in labels.items() if k in by_labels
                )
            elif without_labels is not None:
                group_key = frozenset(
                    (k, v) for k, v in labels.items() if k not in without_labels
                )
            else:
                group_key = frozenset()   # merge everything into one group
            groups.setdefault(group_key, []).append(qr)

        output: list[QueryResult] = []
        for group_labels, group_series in groups.items():
            result_samples: list[Sample] = []

            for t in steps:
                vals = []
                for qr in group_series:
                    v = _get_value_at(qr.samples, t)
                    if v is not None:
                        vals.append(v)
                if vals:
                    result_samples.append(
                        Sample(timestamp=t, value=_apply_agg_fn(fn, vals))
                    )

            series_name = group_series[0].key.name if group_series else ""
            key = MetricKey(name=series_name, labels=group_labels)
            output.append(QueryResult(key=key, samples=result_samples))

        return output

    # --- Histogram functions -------------------------------------------

    def _eval_histogram_quantile(
        self,
        node: FunctionCall,
        start: float,
        end: float,
        step: float,
        steps: list[float],
    ) -> list[QueryResult]:
        """Evaluate ``histogram_quantile(φ, metric{le="..."})``."""
        if len(node.args) < 2:
            raise QueryEvalError("histogram_quantile requires two arguments: (φ, series)")

        phi_node = node.args[0]
        phi = self._eval(phi_node, start, end, step)
        if not isinstance(phi, float):
            raise QueryEvalError("histogram_quantile: first argument must be a scalar φ")
        if not (0.0 < phi <= 1.0):
            raise QueryEvalError(
                f"histogram_quantile: φ must be in (0, 1], got {phi}"
            )

        series_node = node.args[1]
        raw_series = self._eval(series_node, start, end, step)
        if not isinstance(raw_series, list):
            raise QueryEvalError("histogram_quantile: second argument must be a series")

        return _compute_histogram_quantile(phi, raw_series, steps)

    def _histogram_simple(
        self,
        fn: str,
        series_list: list[QueryResult],
        steps: list[float],
    ) -> list[QueryResult]:
        """Implement histogram_count / histogram_sum by summing all bucket values.

        Both functions just sum across all le-bucket series that share the same
        base label set (everything except ``le``).  histogram_count gives you
        the total observation count (the +Inf bucket value); histogram_sum gives
        the sum of observed values.  Since we don't distinguish the two in raw
        storage we return the sum of all bucket values, which is a reasonable
        approximation for the common use-case of understanding scale.

        TODO: distinguish _count and _sum suffixes when the ingest layer starts
        tagging them separately.
        """
        groups: dict[frozenset, list[QueryResult]] = {}
        for qr in series_list:
            base_labels = frozenset(
                (k, v) for k, v in qr.key.labels if k != "le"
            )
            groups.setdefault(base_labels, []).append(qr)

        output: list[QueryResult] = []
        for base_labels, group in groups.items():
            result_samples: list[Sample] = []
            for t in steps:
                total = 0.0
                found = False
                for qr in group:
                    v = _get_value_at(qr.samples, t)
                    if v is not None:
                        total += v
                        found = True
                if found:
                    result_samples.append(Sample(timestamp=t, value=total))

            name = group[0].key.name if group else ""
            key = MetricKey(name=name, labels=base_labels)
            output.append(QueryResult(key=key, samples=result_samples))

        return output

    # ------------------------------------------------------------------
    # Binary operations
    # ------------------------------------------------------------------

    def _eval_binary_op(
        self, node: BinaryOp, start: float, end: float, step: float
    ) -> _EvalResult:
        left = self._eval(node.left, start, end, step)
        right = self._eval(node.right, start, end, step)

        # Scalar + Scalar
        if isinstance(left, float) and isinstance(right, float):
            return _scalar_op(node.op, left, right)

        # Series op Scalar
        if isinstance(left, list) and isinstance(right, float):
            return _series_scalar_op(node.op, left, right, swapped=False)

        # Scalar op Series (commutative cases: +, *, ==, !=)
        if isinstance(left, float) and isinstance(right, list):
            return _series_scalar_op(node.op, right, left, swapped=True)

        # Series op Series — match by label fingerprint
        if isinstance(left, list) and isinstance(right, list):
            return _vector_op(node.op, left, right)

        raise QueryEvalError("Unexpected operand types in binary expression")


# ------------------------------------------------------------------
# Pure helper functions (no engine dependency)
# ------------------------------------------------------------------

def _build_index_matchers(selector: MetricSelector):
    """Convert an AST :class:`MetricSelector` into index LabelMatcher objects."""
    from tsdb.index import LabelMatcher as IndexMatcher
    matchers = []
    if selector.name:
        matchers.append(IndexMatcher(name="__name__", op="=", value=selector.name))
    for m in selector.matchers:
        matchers.append(IndexMatcher(name=m.name, op=m.op, value=m.value))
    return matchers


def _make_steps(start: float, end: float, step: float) -> list[float]:
    """Return evenly-spaced evaluation timestamps in [start, end]."""
    if step <= 0:
        return [start] if start == end else [start, end]
    steps: list[float] = []
    n = 0
    while True:
        t = start + n * step
        if t > end + 1e-9:
            break
        steps.append(t)
        n += 1
    return steps or [start]


def _get_value_at(
    samples: list[Sample], t: float, lookback: float = _INSTANT_LOOKBACK
) -> float | None:
    """Return the value of the most-recent sample at or before *t*.

    Looks back up to *lookback* seconds.  Returns ``None`` when no sample
    is found within the window.
    """
    relevant = [s for s in samples if t - lookback <= s.timestamp <= t]
    if not relevant:
        return None
    return max(relevant, key=lambda s: s.timestamp).value


def _apply_window_fn(fn: str, window: list[Sample], duration: float) -> float | None:
    """Compute a window-function result over a list of samples."""
    if not window:
        return None

    values = [s.value for s in window]

    if fn == "rate":
        if len(window) < 2:
            return None
        dt = window[-1].timestamp - window[0].timestamp
        if dt <= 0:
            return None
        delta = window[-1].value - window[0].value
        if delta < 0:
            delta = window[-1].value   # counter reset: measure from 0
        return delta / dt

    if fn == "irate":
        if len(window) < 2:
            return None
        last, prev = window[-1], window[-2]
        dt = last.timestamp - prev.timestamp
        if dt <= 0:
            return None
        delta = last.value - prev.value
        if delta < 0:
            delta = last.value
        return delta / dt

    if fn == "delta":
        if len(window) < 2:
            return None
        return window[-1].value - window[0].value

    if fn == "increase":
        if len(window) < 2:
            return None
        delta = window[-1].value - window[0].value
        if delta < 0:
            delta = window[-1].value   # counter reset
        return delta

    if fn == "changes":
        # Count the number of times the value changed within the window.
        # Consecutive duplicate values don't count.
        if len(window) < 2:
            return 0.0
        count = 0
        for i in range(1, len(window)):
            if window[i].value != window[i - 1].value:
                count += 1
        return float(count)

    if fn == "resets":
        # Count counter resets — a reset is when the value drops (goes down).
        # Useful for detecting process restarts on monotonic counters.
        # TODO: might want to add a threshold here to filter out tiny floating-point noise
        if len(window) < 2:
            return 0.0
        count = 0
        for i in range(1, len(window)):
            if window[i].value < window[i - 1].value:
                count += 1
        return float(count)

    if fn == "avg_over_time":
        return sum(values) / len(values)

    if fn == "min_over_time":
        return min(values)

    if fn == "max_over_time":
        return max(values)

    if fn == "sum_over_time":
        return sum(values)

    raise QueryEvalError(f"Unknown window function: {fn!r}")


def _apply_agg_fn(fn: str, values: list[float]) -> float:
    if fn == "sum":
        return sum(values)
    if fn == "avg":
        return sum(values) / len(values)
    if fn == "min":
        return min(values)
    if fn == "max":
        return max(values)
    if fn == "count":
        return float(len(values))
    raise QueryEvalError(f"Unknown aggregation function: {fn!r}")


def _scalar_op(op: str, a: float, b: float) -> float:
    if op == '+':   return a + b
    if op == '-':   return a - b
    if op == '*':   return a * b
    if op == '/':   return a / b if b != 0.0 else math.nan
    if op == '==':  return 1.0 if a == b else 0.0
    if op == '!=':  return 1.0 if a != b else 0.0
    if op == '>':   return 1.0 if a > b else 0.0
    if op == '<':   return 1.0 if a < b else 0.0
    if op == '>=':  return 1.0 if a >= b else 0.0
    if op == '<=':  return 1.0 if a <= b else 0.0
    raise QueryEvalError(f"Unknown binary operator: {op!r}")


def _series_scalar_op(
    op: str,
    series: list[QueryResult],
    scalar: float,
    swapped: bool = False,
) -> list[QueryResult]:
    """Apply *op* between every sample in *series* and the constant *scalar*."""
    result: list[QueryResult] = []
    for qr in series:
        new_samples: list[Sample] = []
        for s in qr.samples:
            a, b = (scalar, s.value) if swapped else (s.value, scalar)
            val = _scalar_op(op, a, b)
            new_samples.append(Sample(timestamp=s.timestamp, value=val))
        result.append(QueryResult(key=qr.key, samples=new_samples))
    return result


def _compute_histogram_quantile(
    phi: float,
    series_list: list[QueryResult],
    steps: list[float],
) -> list[QueryResult]:
    """Pure helper: compute φ-quantile from a set of histogram bucket series.

    Algorithm
    ---------
    1. Group series by all labels *except* ``le``.
    2. At each evaluation step, read the bucket counts and sort by le boundary.
    3. Find the two adjacent buckets that straddle ``φ * total`` observations
       and linearly interpolate the bucket boundary.
    4. Return one result series per histogram group, without the ``le`` label.
    """
    # Group by base labels (drop le).
    groups: dict[frozenset, list[QueryResult]] = {}
    for qr in series_list:
        base_labels = frozenset(
            (k, v) for k, v in qr.key.labels if k != "le"
        )
        groups.setdefault(base_labels, []).append(qr)

    output: list[QueryResult] = []
    for base_labels, group in groups.items():
        result_samples: list[Sample] = []

        for t in steps:
            # Build (le_value, cumulative_count) pairs for this timestamp.
            buckets: list[tuple[float, float]] = []
            for qr in group:
                le_str = dict(qr.key.labels).get("le", "")
                if le_str == "":
                    continue
                try:
                    le_val = math.inf if le_str == "+Inf" else float(le_str)
                except ValueError:
                    continue
                v = _get_value_at(qr.samples, t)
                if v is not None:
                    buckets.append((le_val, v))

            if not buckets:
                continue

            # Sort ascending, +Inf last.
            buckets.sort(key=lambda x: (math.isinf(x[0]), x[0]))

            total = buckets[-1][1]  # cumulative count in the +Inf bucket
            if total <= 0:
                continue

            target = phi * total

            # Find the straddling bucket pair.
            lower_le = 0.0
            lower_count = 0.0
            quantile_value: float | None = None

            for le_val, cum_count in buckets:
                if cum_count >= target:
                    # Interpolate between [lower_le, le_val].
                    bucket_width = le_val - lower_le
                    bucket_count = cum_count - lower_count
                    if bucket_count <= 0 or math.isinf(bucket_width):
                        quantile_value = lower_le
                    else:
                        frac = (target - lower_count) / bucket_count
                        quantile_value = lower_le + frac * bucket_width
                    break
                lower_le = le_val
                lower_count = cum_count

            if quantile_value is not None:
                result_samples.append(Sample(timestamp=t, value=quantile_value))

        name = group[0].key.name if group else ""
        key = MetricKey(name=name, labels=base_labels)
        output.append(QueryResult(key=key, samples=result_samples))

    return output


def _vector_op(
    op: str,
    left: list[QueryResult],
    right: list[QueryResult],
) -> list[QueryResult]:
    """Element-wise operation between two vectors, matched by label fingerprint."""
    right_index: dict[frozenset, QueryResult] = {qr.key.labels: qr for qr in right}

    result: list[QueryResult] = []
    for lqr in left:
        rqr = right_index.get(lqr.key.labels)
        if rqr is None:
            continue   # no matching series on right side; skip (one-to-one matching)

        # Index right samples by timestamp for O(1) lookup.
        right_by_ts: dict[float, float] = {s.timestamp: s.value for s in rqr.samples}

        new_samples: list[Sample] = []
        for s in lqr.samples:
            rv = right_by_ts.get(s.timestamp)
            if rv is not None:
                val = _scalar_op(op, s.value, rv)
                new_samples.append(Sample(timestamp=s.timestamp, value=val))

        result.append(QueryResult(key=lqr.key, samples=new_samples))

    return result
