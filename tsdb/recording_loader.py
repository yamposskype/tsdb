"""Load recording rules from plain Python dicts (parsed from YAML or JSON).

Expected config shape::

    {
        "groups": [
            {
                "name": "aggregations",
                "interval": "1m",          # optional, default 60s
                "rules": [
                    {
                        "record": "job:requests:rate5m",
                        "expr": "rate(requests_total{job='api'}[5m])",
                        "labels": {"env": "prod"}  # optional
                    }
                ]
            }
        ]
    }

The ``interval`` field (both at group level and per-rule) accepts a duration
string: plain seconds (``"30s"``), minutes (``"5m"``), or hours (``"1h"``).
A bare number is treated as seconds.

Per-rule interval is not part of the standard Prometheus format, but we
support it anyway in case someone needs finer control.  When absent the
group interval is used.
"""

from __future__ import annotations

import re

from tsdb.recording import RecordingRule, RecordingRuleGroup

_DURATION_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[smh]?)$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "": 1}

_DEFAULT_INTERVAL = 60.0


def _parse_duration(raw) -> float:
    """Convert a duration string or number to seconds."""
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


def load_recording_rules_from_dict(config: dict) -> list[RecordingRuleGroup]:
    """Build a list of :class:`~tsdb.recording.RecordingRuleGroup` from a config dict.

    Parameters
    ----------
    config:
        Dict with a ``"groups"`` key, each entry describing one rule group.

    Returns
    -------
    list[RecordingRuleGroup]
        Parsed and validated group objects.

    Raises
    ------
    ValueError
        When a required field is missing or a value cannot be parsed.
    """
    raw_groups = config.get("groups", [])
    if not isinstance(raw_groups, list):
        raise ValueError("config['groups'] must be a list")

    groups: list[RecordingRuleGroup] = []

    for gi, raw_group in enumerate(raw_groups):
        try:
            group_name = raw_group["name"]
        except KeyError:
            raise ValueError(f"Group #{gi}: missing required field 'name'")

        group_interval = _parse_duration(raw_group.get("interval", _DEFAULT_INTERVAL))

        raw_rules = raw_group.get("rules", [])
        if not isinstance(raw_rules, list):
            raise ValueError(f"Group {group_name!r}: 'rules' must be a list")

        rules: list[RecordingRule] = []

        for ri, raw_rule in enumerate(raw_rules):
            try:
                record_name = raw_rule["record"]
                expr = raw_rule["expr"]
            except KeyError as exc:
                raise ValueError(
                    f"Group {group_name!r}, rule #{ri}: missing required field {exc}"
                ) from exc

            # Per-rule interval falls back to the group interval.
            rule_interval_raw = raw_rule.get("interval", None)
            if rule_interval_raw is not None:
                rule_interval = _parse_duration(rule_interval_raw)
            else:
                rule_interval = group_interval

            labels = dict(raw_rule.get("labels") or {})

            rules.append(
                RecordingRule(
                    name=record_name,
                    expr=expr,
                    interval=rule_interval,
                    labels=labels,
                )
            )

        groups.append(
            RecordingRuleGroup(
                name=group_name,
                interval=group_interval,
                rules=rules,
            )
        )

    return groups
