"""Exceptions raised by the query language parser and evaluator."""


class QueryParseError(Exception):
    """Raised when the query string cannot be tokenized or parsed."""


class QueryEvalError(Exception):
    """Raised when a successfully parsed query cannot be evaluated."""
