"""AST node types for the query language.

Each dataclass represents one syntactic construct.  The parser builds a
tree of these; the evaluator walks it to produce :class:`QueryResult` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LabelMatcher:
    """A single label filter: name op value.

    *op* is one of ``=``, ``!=``, ``=~``, ``!~`` — the same operators
    supported by the storage index.
    """
    name: str
    op: str
    value: str


@dataclass
class MetricSelector:
    """Selects one or more time series by metric name and label matchers.

    When *name* is an empty string the selector matches only on labels.
    """
    name: str
    matchers: list[LabelMatcher] = field(default_factory=list)


@dataclass
class RangeQuery:
    """Wraps a :class:`MetricSelector` with a sliding time window.

    *range_duration* is the window width in seconds.
    *offset* shifts the window backwards in time by the given number of seconds.
    """
    selector: MetricSelector
    range_duration: float       # seconds
    offset: float = 0.0         # seconds; positive = look further back


@dataclass
class FunctionCall:
    """A function applied to one or more sub-expressions.

    *by* restricts aggregation to the listed label names.
    *without* aggregates over all labels except the listed ones.
    Both are ``None`` when no grouping modifier is present.
    """
    function: str
    args: list = field(default_factory=list)
    by: list[str] | None = None
    without: list[str] | None = None


@dataclass
class BinaryOp:
    """A binary operation between two sub-expressions.

    *op* is one of: ``+``, ``-``, ``*``, ``/``,
    ``==``, ``!=``, ``>``, ``<``, ``>=``, ``<=``.

    *bool_modifier* (PromQL ``bool``) returns 0/1 rather than filtering.
    """
    op: str
    left: object
    right: object
    bool_modifier: bool = False


@dataclass
class NumberLiteral:
    """A numeric literal used as a scalar operand."""
    value: float
