"""tsdb.ql — PromQL-inspired query language for tsdb.

Quick start::

    from tsdb.ql import parse_query, QueryEvaluator

    # Parse only
    ast = parse_query("rate(requests_total{job='api'}[5m])")

    # Parse and evaluate
    evaluator = QueryEvaluator(engine)
    results   = evaluator.evaluate(
        "sum(http_requests[1m]) by (job)",
        start=1_700_000_000.0,
        end=1_700_003_600.0,
        step=60.0,
    )
"""

from tsdb.ql.evaluator import QueryEvaluator
from tsdb.ql.parser import Parser, parse_query

__all__ = ["Parser", "parse_query", "QueryEvaluator"]
