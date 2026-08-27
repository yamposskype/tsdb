"""Load alert rules from plain Python dicts (e.g. parsed from YAML or JSON).

Expected dict shape::

    {
        "name": "high_cpu",
        "expr": "cpu_usage{job='api'}",
        "condition": "gt",
        "threshold": 90,
        "for": "5m",          # optional, default "0s"
        "severity": "warning", # optional, default "warning"
        "labels": {},          # optional
        "annotations": {}      # optional
    }

The ``for`` field accepts a duration string: plain seconds (``"30s"``), minutes
(``"5m"``), or hours (``"1h"``).  A bare integer is also accepted and treated as
seconds.
"""

from __future__ import annotations

import re

from tsdb.alerting.rule import AlertRule, AlertSeverity

_DURATION_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[smh]?)$")

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "": 1}


def _parse_duration(raw) -> float:
    """Convert a duration string or number to seconds.

    Supported formats: ``"30s"``, ``"5m"``, ``"1h"``, or a bare number.
    """
    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    m = _DURATION_RE.match(text)
    if not m:
        raise ValueError(
            f"Cannot parse duration {text!r}; expected a number optionally "
            f"followed by 's', 'm', or 'h' (e.g. '30s', '5m', '1h')"
        )
    value = float(m.group("value"))
    unit = m.group("unit")
    return value * _UNIT_SECONDS[unit]


def load_rules(config: list[dict]) -> list[AlertRule]:
    """Build a list of :class:`~tsdb.alerting.rule.AlertRule` from raw dicts.

    Parameters
    ----------
    config:
        A list of dicts, each describing one alert rule.

    Returns
    -------
    list[AlertRule]
        Parsed and validated rule objects.

    Raises
    ------
    ValueError
        When a required field is missing or a value cannot be parsed.
    """
    rules: list[AlertRule] = []

    for i, raw in enumerate(config):
        try:
            name = raw["name"]
            expr = raw["expr"]
            condition = raw["condition"]
            threshold = float(raw["threshold"])
        except KeyError as exc:
            raise ValueError(
                f"Rule #{i}: missing required field {exc}"
            ) from exc

        for_duration = _parse_duration(raw.get("for", "0s"))

        severity_raw = raw.get("severity", "warning")
        try:
            severity = AlertSeverity(severity_raw)
        except ValueError:
            valid = [s.value for s in AlertSeverity]
            raise ValueError(
                f"Rule {name!r}: unknown severity {severity_raw!r}; valid: {valid}"
            )

        labels = dict(raw.get("labels") or {})
        annotations = dict(raw.get("annotations") or {})

        rules.append(
            AlertRule(
                name=name,
                expr=expr,
                condition=condition,
                threshold=threshold,
                for_duration=for_duration,
                severity=severity,
                labels=labels,
                annotations=annotations,
            )
        )

    return rules
