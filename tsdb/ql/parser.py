"""Recursive-descent parser for the PromQL-inspired query language.

Grammar (simplified, precedence low → high)::

    query            : expr EOF
    expr             : comparison_expr
    comparison_expr  : additive_expr (comparison_op additive_expr)?
    additive_expr    : multiplicative_expr (('+' | '-') multiplicative_expr)*
    multiplicative_expr : primary (('*' | '/') primary)*
    primary          : LPAREN expr RPAREN
                     | NUMBER
                     | unary_minus
                     | function_call
                     | selector_expr

    function_call    : FUNCTION LPAREN arg_list RPAREN (by_clause | without_clause)?
    by_clause        : BY LPAREN label_list RPAREN
    without_clause   : WITHOUT LPAREN label_list RPAREN
    arg_list         : expr (',' expr)*

    selector_expr    : IDENTIFIER (LBRACE matcher_list? RBRACE)?
                       (LBRACKET DURATION RBRACKET (OFFSET DURATION)?)?

    matcher_list     : matcher (',' matcher)*
    matcher          : IDENTIFIER op STRING
    op               : EQ | NEQ | REGEX_MATCH | REGEX_NOT_MATCH

    label_list       : IDENTIFIER (',' IDENTIFIER)*
    comparison_op    : OP where value in {'==', '!=', '>=', '<=', '>', '<'}
"""

from __future__ import annotations

from tsdb.ql.ast import (
    BinaryOp,
    FunctionCall,
    LabelMatcher,
    MetricSelector,
    NumberLiteral,
    RangeQuery,
)
from tsdb.ql.errors import QueryParseError
from tsdb.ql.tokenizer import Token, TokenType

# Binary operator sets by precedence tier
_COMPARISON_OPS: frozenset[str] = frozenset({'==', '!=', '>=', '<=', '>', '<'})
_ADDITIVE_OPS: frozenset[str] = frozenset({'+', '-'})
_MULTIPLICATIVE_OPS: frozenset[str] = frozenset({'*', '/'})

# Duration unit → seconds
_DURATION_UNITS: dict[str, float] = {
    'ms': 0.001,
    's':  1.0,
    'm':  60.0,
    'h':  3600.0,
    'd':  86400.0,
    'w':  604800.0,
}

# Label matcher operator tokens
_MATCHER_OP_TYPES: frozenset[TokenType] = frozenset({
    TokenType.EQ,
    TokenType.NEQ,
    TokenType.REGEX_MATCH,
    TokenType.REGEX_NOT_MATCH,
})


class Parser:
    """Recursive-descent parser.

    Usage::

        tokens = Tokenizer(query_string).tokenize()
        ast    = Parser(tokens).parse()
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self):
        """Parse the full token stream and return the root AST node."""
        node = self._parse_expr()
        if self._current.type != TokenType.EOF:
            raise QueryParseError(
                f"Unexpected token after expression: "
                f"{self._current.type.name} ({self._current.value!r})"
            )
        return node

    # ------------------------------------------------------------------
    # Expression tiers (low precedence → high precedence)
    # ------------------------------------------------------------------

    def _parse_expr(self):
        return self._parse_comparison()

    def _parse_comparison(self):
        left = self._parse_additive()
        # Only one comparison operator is allowed (no chaining)
        while (
            self._current.type == TokenType.OP
            and self._current.value in _COMPARISON_OPS
        ) or self._current.type == TokenType.NEQ:
            if self._current.type == TokenType.NEQ:
                op = '!='
            else:
                op = self._current.value
            self._advance()
            right = self._parse_additive()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def _parse_additive(self):
        left = self._parse_multiplicative()
        while (
            self._current.type == TokenType.OP
            and self._current.value in _ADDITIVE_OPS
        ):
            op = self._advance().value
            right = self._parse_multiplicative()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def _parse_multiplicative(self):
        left = self._parse_primary()
        while (
            self._current.type == TokenType.OP
            and self._current.value in _MULTIPLICATIVE_OPS
        ):
            op = self._advance().value
            right = self._parse_primary()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def _parse_primary(self):
        tok = self._current

        # Parenthesised sub-expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            node = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return node

        # Numeric literal
        if tok.type == TokenType.NUMBER:
            self._advance()
            return NumberLiteral(value=float(tok.value))

        # Unary minus: -expr
        if tok.type == TokenType.OP and tok.value == '-':
            self._advance()
            inner = self._parse_primary()
            if isinstance(inner, NumberLiteral):
                return NumberLiteral(value=-inner.value)
            # Negate a series by multiplying by -1
            return BinaryOp(op='*', left=NumberLiteral(value=-1.0), right=inner)

        # Function call: FUNCTION ( args ) [by/without (...)]
        if tok.type == TokenType.FUNCTION:
            return self._parse_function_call()

        # Metric selector: identifier { matchers } [ duration ] offset duration
        if tok.type in (TokenType.IDENTIFIER, TokenType.METRIC_NAME):
            return self._parse_selector_expr()

        raise QueryParseError(
            f"Unexpected token in expression: {tok.type.name} ({tok.value!r})"
        )

    # ------------------------------------------------------------------
    # Function calls
    # ------------------------------------------------------------------

    def _parse_function_call(self) -> FunctionCall:
        func_tok = self._expect(TokenType.FUNCTION)
        self._expect(TokenType.LPAREN)

        args = []
        if self._current.type != TokenType.RPAREN:
            args.append(self._parse_expr())
            while self._current.type == TokenType.COMMA:
                self._advance()
                args.append(self._parse_expr())

        self._expect(TokenType.RPAREN)

        by_labels: list[str] | None = None
        without_labels: list[str] | None = None

        if self._current.type == TokenType.BY:
            self._advance()
            by_labels = self._parse_label_list()
        elif self._current.type == TokenType.WITHOUT:
            self._advance()
            without_labels = self._parse_label_list()

        return FunctionCall(
            function=func_tok.value,
            args=args,
            by=by_labels,
            without=without_labels,
        )

    # ------------------------------------------------------------------
    # Metric selector  (identifier { matchers } [ duration ] offset dur)
    # ------------------------------------------------------------------

    def _parse_selector_expr(self):
        name_tok = self._advance()
        name = name_tok.value

        matchers: list[LabelMatcher] = []
        if self._current.type == TokenType.LBRACE:
            matchers = self._parse_matchers()

        selector = MetricSelector(name=name, matchers=matchers)

        # Range window?
        if self._current.type == TokenType.LBRACKET:
            self._advance()
            dur_tok = self._expect(TokenType.DURATION)
            duration_secs = _parse_duration(dur_tok.value)
            self._expect(TokenType.RBRACKET)

            offset_secs = 0.0
            if self._current.type == TokenType.OFFSET:
                self._advance()
                off_tok = self._expect(TokenType.DURATION)
                offset_secs = _parse_duration(off_tok.value)

            return RangeQuery(
                selector=selector,
                range_duration=duration_secs,
                offset=offset_secs,
            )

        return selector

    # ------------------------------------------------------------------
    # Label matchers  { name op "value", ... }
    # ------------------------------------------------------------------

    def _parse_matchers(self) -> list[LabelMatcher]:
        self._expect(TokenType.LBRACE)
        matchers: list[LabelMatcher] = []

        while self._current.type != TokenType.RBRACE:
            if self._current.type == TokenType.EOF:
                raise QueryParseError("Unterminated label matcher — missing '}'")

            label_tok = self._expect(TokenType.IDENTIFIER)

            op_tok = self._current
            if op_tok.type not in _MATCHER_OP_TYPES:
                raise QueryParseError(
                    f"Expected label matcher operator (=, !=, =~, !~), "
                    f"got {op_tok.type.name} ({op_tok.value!r})"
                )
            op_str = _matcher_op_str(op_tok.type)
            self._advance()

            val_tok = self._expect(TokenType.STRING)
            matchers.append(
                LabelMatcher(name=label_tok.value, op=op_str, value=val_tok.value)
            )

            if self._current.type == TokenType.COMMA:
                self._advance()

        self._expect(TokenType.RBRACE)
        return matchers

    # ------------------------------------------------------------------
    # Label lists  ( label1, label2, ... )
    # ------------------------------------------------------------------

    def _parse_label_list(self) -> list[str]:
        self._expect(TokenType.LPAREN)
        labels: list[str] = []
        while self._current.type == TokenType.IDENTIFIER:
            labels.append(self._advance().value)
            if self._current.type == TokenType.COMMA:
                self._advance()
        self._expect(TokenType.RPAREN)
        return labels

    # ------------------------------------------------------------------
    # Token stream helpers
    # ------------------------------------------------------------------

    @property
    def _current(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return tok

    def _expect(self, ttype: TokenType) -> Token:
        tok = self._current
        if tok.type != ttype:
            raise QueryParseError(
                f"Expected {ttype.name}, got {tok.type.name} ({tok.value!r})"
            )
        return self._advance()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _parse_duration(s: str) -> float:
    """Convert a duration string like '5m' or '1h30s' (single unit) to seconds."""
    for suffix, multiplier in sorted(_DURATION_UNITS.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suffix):
            num_part = s[: -len(suffix)]
            try:
                return float(num_part) * multiplier
            except ValueError:
                raise QueryParseError(f"Invalid duration number in {s!r}")
    raise QueryParseError(f"Unknown duration unit in {s!r}")


def _matcher_op_str(ttype: TokenType) -> str:
    return {
        TokenType.EQ: '=',
        TokenType.NEQ: '!=',
        TokenType.REGEX_MATCH: '=~',
        TokenType.REGEX_NOT_MATCH: '!~',
    }[ttype]


def parse_query(query_str: str):
    """Tokenize and parse *query_str*, returning the root AST node.

    Raises :class:`~tsdb.ql.errors.QueryParseError` on syntax errors.
    """
    from tsdb.ql.tokenizer import Tokenizer
    tokens = Tokenizer(query_str).tokenize()
    return Parser(tokens).parse()
