"""Inverted label index for fast metric lookup by label matchers.

The index maps label_name -> label_value -> set[MetricKey], so any query that
specifies one or more label constraints can jump straight to the candidate set
rather than scanning all stored series.

Supported matcher operators
  =    exact match
  !=   not equal
  =~   regex full-match (re.fullmatch)
  !~   regex not full-match
"""

import re
import threading
from dataclasses import dataclass

from tsdb.types import MetricKey


@dataclass
class LabelMatcher:
    """A single label filter used to select series from the index."""

    name: str
    value: str
    op: str  # one of: =, !=, =~, !~

    def matches(self, label_value: str) -> bool:
        if self.op == "=":
            return label_value == self.value
        if self.op == "!=":
            return label_value != self.value
        if self.op == "=~":
            return re.fullmatch(self.value, label_value) is not None
        if self.op == "!~":
            return re.fullmatch(self.value, label_value) is None
        raise ValueError(f"Unknown matcher operator: {self.op!r}")


# ---------------------------------------------------------------------------
# Matcher string parser
# ---------------------------------------------------------------------------

_MATCHER_RE = re.compile(r'\s*(\w+)\s*(!=|=~|!~|=)\s*"([^"]*)"\s*,?')


def parse_matchers(label_str: str) -> list[LabelMatcher]:
    """Parse a Prometheus-style label selector string into LabelMatchers.

    Accepted forms::

        {job="web", env=~"prod.*"}
        {host!="db1", region!~"us-.*"}
        job="web"           # curly braces are optional

    Returns an empty list for an empty or bare ``{}`` selector.
    """
    label_str = label_str.strip()
    if label_str.startswith("{"):
        label_str = label_str[1:]
    if label_str.endswith("}"):
        label_str = label_str[:-1]
    label_str = label_str.strip()

    if not label_str:
        return []

    matchers: list[LabelMatcher] = []
    for m in _MATCHER_RE.finditer(label_str):
        matchers.append(LabelMatcher(name=m.group(1), op=m.group(2), value=m.group(3)))
    return matchers


# ---------------------------------------------------------------------------
# InvertedIndex
# ---------------------------------------------------------------------------

class InvertedIndex:
    """Thread-safe inverted label index.

    Internal layout::

        _index[label_name][label_value] = {MetricKey, ...}

    The special label name ``__name__`` stores the metric name so
    ``find_by_name`` is a plain dict lookup rather than a scan.

    ``find`` applies multiple matchers with AND semantics:
      - ``=``  matchers seed / narrow the result via dict lookup (cheapest)
      - ``=~`` matchers union matching value buckets then intersect
      - ``!=`` and ``!~`` matchers compute keys to exclude from the result
    """

    def __init__(self) -> None:
        self._index: dict[str, dict[str, set[MetricKey]]] = {}
        self._all_keys: set[MetricKey] = set()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, key: MetricKey) -> None:
        """Index all labels (and the metric name) of *key*."""
        with self._lock:
            if key in self._all_keys:
                return
            self._all_keys.add(key)

            # Index every label pair
            for label_name, label_value in key.labels:
                self._index.setdefault(label_name, {}).setdefault(label_value, set()).add(key)

            # Index metric name under the reserved __name__ label
            self._index.setdefault("__name__", {}).setdefault(key.name, set()).add(key)

    def remove(self, key: MetricKey) -> None:
        """Remove *key* from every label bucket it occupies."""
        with self._lock:
            self._all_keys.discard(key)

            for label_name, label_value in key.labels:
                values_map = self._index.get(label_name)
                if values_map is None:
                    continue
                bucket = values_map.get(label_value)
                if bucket is not None:
                    bucket.discard(key)
                    if not bucket:
                        del values_map[label_value]
                if not values_map:
                    del self._index[label_name]

            name_map = self._index.get("__name__")
            if name_map is not None:
                bucket = name_map.get(key.name)
                if bucket is not None:
                    bucket.discard(key)
                    if not bucket:
                        del name_map[key.name]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def find(self, matchers: list[LabelMatcher]) -> set[MetricKey]:
        """Return all MetricKeys that satisfy every matcher (AND semantics).

        Processing order:
        1. ``=``  — dict lookup, intersect into running result (fastest narrowing)
        2. ``=~`` — regex over value buckets, union matching keys, intersect
        3. Start from all keys if no positive matchers constrained the result
        4. ``!=``  — remove keys whose label value equals the matcher value
        5. ``!~``  — remove keys whose label value regex-matches
        """
        with self._lock:
            if not matchers:
                return set(self._all_keys)

            eq_matchers  = [m for m in matchers if m.op == "="]
            inc_re       = [m for m in matchers if m.op == "=~"]
            neq_matchers = [m for m in matchers if m.op == "!="]
            exc_re       = [m for m in matchers if m.op == "!~"]

            result: set[MetricKey] | None = None

            # Exact-match matchers — most selective, use index directly
            for matcher in eq_matchers:
                values_map = self._index.get(matcher.name, {})
                candidates = set(values_map.get(matcher.value, set()))
                result = candidates if result is None else result & candidates
                if not result:
                    return set()

            # Regex-include matchers — union matching buckets then intersect
            for matcher in inc_re:
                values_map = self._index.get(matcher.name, {})
                matched: set[MetricKey] = set()
                for label_val, keys in values_map.items():
                    if re.fullmatch(matcher.value, label_val) is not None:
                        matched |= keys
                result = matched if result is None else result & matched
                if not result:
                    return set()

            # No positive constraints at all — start from the full set
            if result is None:
                result = set(self._all_keys)

            # Exclusion pass — build the set to remove, then subtract once
            exclusions: set[MetricKey] = set()

            for matcher in neq_matchers:
                values_map = self._index.get(matcher.name, {})
                exclusions |= set(values_map.get(matcher.value, set()))

            for matcher in exc_re:
                values_map = self._index.get(matcher.name, {})
                for label_val, keys in values_map.items():
                    if re.fullmatch(matcher.value, label_val) is not None:
                        exclusions |= keys

            return result - exclusions

    def find_by_name(self, name: str) -> set[MetricKey]:
        """Return all series with this metric name."""
        with self._lock:
            name_map = self._index.get("__name__", {})
            return set(name_map.get(name, set()))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def all_label_names(self) -> list[str]:
        """Sorted list of every indexed label name (``__name__`` excluded)."""
        with self._lock:
            return sorted(n for n in self._index if n != "__name__")

    def all_label_values(self, name: str) -> list[str]:
        """Sorted list of every indexed value for the given label name."""
        with self._lock:
            return sorted(self._index.get(name, {}).keys())

    def size(self) -> int:
        """Number of currently indexed series."""
        with self._lock:
            return len(self._all_keys)
