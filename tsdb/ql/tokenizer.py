"""Query language tokenizer.

Converts a raw query string into a flat list of typed tokens that the
parser can consume one at a time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from tsdb.ql.errors import QueryParseError


class TokenType(Enum):
    METRIC_NAME = auto()        # reserved; parser uses IDENTIFIER contextually
    LBRACE = auto()             # {
    RBRACE = auto()             # }
    LBRACKET = auto()           # [
    RBRACKET = auto()           # ]
    LPAREN = auto()             # (
    RPAREN = auto()             # )
    COMMA = auto()              # ,
    EQ = auto()                 # =   (label matcher)
    NEQ = auto()                # !=  (label matcher / binary comparison)
    REGEX_MATCH = auto()        # =~
    REGEX_NOT_MATCH = auto()    # !~
    STRING = auto()             # "…"
    NUMBER = auto()             # 42 / 3.14
    DURATION = auto()           # 5m / 1h / 30s
    IDENTIFIER = auto()         # bare word — metric name, label name
    FUNCTION = auto()           # recognised function name
    BY = auto()                 # by keyword
    WITHOUT = auto()            # without keyword
    OFFSET = auto()             # offset keyword
    OP = auto()                 # arithmetic / comparison: + - * / == > < >= <=
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str


# -----------------------------------------------------------------------
# Static lookup tables
# -----------------------------------------------------------------------

_KEYWORDS: dict[str, TokenType] = {
    "by": TokenType.BY,
    "without": TokenType.WITHOUT,
    "offset": TokenType.OFFSET,
}

_KNOWN_FUNCTIONS: frozenset[str] = frozenset({
    "rate", "irate", "delta", "increase",
    "changes", "resets",
    "sum", "avg", "min", "max", "count",
    "avg_over_time", "min_over_time", "max_over_time", "sum_over_time",
    "histogram_quantile", "histogram_count", "histogram_sum",
})

_DURATION_RE = re.compile(r'^(\d+(?:\.\d+)?)(ms|s|m|h|d|w)$')


class Tokenizer:
    """Converts *text* into a list of :class:`Token` objects.

    Call :meth:`tokenize` once; it returns the complete token list including
    a trailing ``EOF`` sentinel.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        self._pos = 0
        self._tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        while self._pos < len(self._text):
            self._skip_whitespace()
            if self._pos >= len(self._text):
                break
            ch = self._text[self._pos]

            if ch == '"':
                self._read_string()
            elif ch.isdigit():
                self._read_number_or_duration()
            elif ch.isalpha() or ch == '_':
                self._read_identifier()
            elif ch in ('=', '!', '>', '<', '+', '-', '*', '/'):
                self._read_operator()
            elif ch == '{':
                self._emit(TokenType.LBRACE, '{'); self._pos += 1
            elif ch == '}':
                self._emit(TokenType.RBRACE, '}'); self._pos += 1
            elif ch == '[':
                self._emit(TokenType.LBRACKET, '['); self._pos += 1
            elif ch == ']':
                self._emit(TokenType.RBRACKET, ']'); self._pos += 1
            elif ch == '(':
                self._emit(TokenType.LPAREN, '('); self._pos += 1
            elif ch == ')':
                self._emit(TokenType.RPAREN, ')'); self._pos += 1
            elif ch == ',':
                self._emit(TokenType.COMMA, ','); self._pos += 1
            else:
                raise QueryParseError(
                    f"Unexpected character {ch!r} at position {self._pos}"
                )

        self._emit(TokenType.EOF, '')
        return self._tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, ttype: TokenType, value: str) -> None:
        self._tokens.append(Token(ttype, value))

    def _skip_whitespace(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _read_string(self) -> None:
        self._pos += 1  # skip opening "
        start = self._pos
        while self._pos < len(self._text) and self._text[self._pos] != '"':
            if self._text[self._pos] == '\\':
                self._pos += 1  # skip the escaped character
            self._pos += 1
        if self._pos >= len(self._text):
            raise QueryParseError("Unterminated string literal")
        value = self._text[start:self._pos]
        self._pos += 1  # skip closing "
        self._emit(TokenType.STRING, value)

    def _read_number_or_duration(self) -> None:
        start = self._pos
        while self._pos < len(self._text) and (
            self._text[self._pos].isdigit() or self._text[self._pos] == '.'
        ):
            self._pos += 1

        # Absorb any trailing alphabetic suffix (could be a duration unit).
        suffix_start = self._pos
        while self._pos < len(self._text) and self._text[self._pos].isalpha():
            self._pos += 1

        raw = self._text[start:self._pos]

        if self._pos > suffix_start:
            m = _DURATION_RE.match(raw)
            if not m:
                raise QueryParseError(f"Invalid duration or number literal: {raw!r}")
            self._emit(TokenType.DURATION, raw)
        else:
            self._emit(TokenType.NUMBER, raw)

    def _read_identifier(self) -> None:
        start = self._pos
        while self._pos < len(self._text) and (
            self._text[self._pos].isalnum() or self._text[self._pos] == '_'
        ):
            self._pos += 1
        word = self._text[start:self._pos]

        if word in _KEYWORDS:
            self._emit(_KEYWORDS[word], word)
        elif word in _KNOWN_FUNCTIONS:
            self._emit(TokenType.FUNCTION, word)
        else:
            self._emit(TokenType.IDENTIFIER, word)

    def _read_operator(self) -> None:
        ch = self._text[self._pos]
        nxt = self._text[self._pos + 1] if self._pos + 1 < len(self._text) else ''

        if ch == '=' and nxt == '~':
            self._emit(TokenType.REGEX_MATCH, '=~'); self._pos += 2
        elif ch == '=' and nxt == '=':
            self._emit(TokenType.OP, '=='); self._pos += 2
        elif ch == '=':
            self._emit(TokenType.EQ, '='); self._pos += 1
        elif ch == '!' and nxt == '=':
            self._emit(TokenType.NEQ, '!='); self._pos += 2
        elif ch == '!' and nxt == '~':
            self._emit(TokenType.REGEX_NOT_MATCH, '!~'); self._pos += 2
        elif ch == '!' :
            raise QueryParseError(f"Unexpected '!' at position {self._pos} (did you mean != or !~?)")
        elif ch == '>' and nxt == '=':
            self._emit(TokenType.OP, '>='); self._pos += 2
        elif ch == '>':
            self._emit(TokenType.OP, '>'); self._pos += 1
        elif ch == '<' and nxt == '=':
            self._emit(TokenType.OP, '<='); self._pos += 2
        elif ch == '<':
            self._emit(TokenType.OP, '<'); self._pos += 1
        elif ch == '+':
            self._emit(TokenType.OP, '+'); self._pos += 1
        elif ch == '-':
            self._emit(TokenType.OP, '-'); self._pos += 1
        elif ch == '*':
            self._emit(TokenType.OP, '*'); self._pos += 1
        elif ch == '/':
            self._emit(TokenType.OP, '/'); self._pos += 1
        else:
            raise QueryParseError(
                f"Unexpected operator character {ch!r} at position {self._pos}"
            )
