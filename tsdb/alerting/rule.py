"""Alert rule definitions.

An AlertRule describes a condition to monitor: it holds a PromQL-like
expression, a comparison operator, a numeric threshold, and optional metadata
(labels and annotations) that are attached to every alert the rule produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AlertRule:
    """A single alerting rule.

    Attributes
    ----------
    name:
        Unique rule identifier.
    expr:
        PromQL-like query expression whose result is compared against the
        threshold.
    condition:
        Comparison operator applied to the query result and the threshold.
        One of ``"gt"``, ``"lt"``, ``"gte"``, ``"lte"``, ``"eq"``, ``"neq"``.
    threshold:
        The numeric value the query result is compared against.
    for_duration:
        Number of seconds the condition must hold continuously before the alert
        transitions from PENDING to FIRING.  Zero means fire immediately.
    severity:
        How serious the alert is.
    labels:
        Extra key-value pairs attached to every alert instance produced by
        this rule.  Useful for routing (e.g. team, service).
    annotations:
        Human-readable key-value pairs (e.g. "summary", "description") that
        are not used for routing but are shown in dashboards and notifications.
    """

    name: str
    expr: str
    condition: str
    threshold: float
    for_duration: float = 0.0
    severity: AlertSeverity = AlertSeverity.WARNING
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_conditions = {"gt", "lt", "gte", "lte", "eq", "neq"}
        if self.condition not in valid_conditions:
            raise ValueError(
                f"Invalid condition {self.condition!r}; must be one of {sorted(valid_conditions)}"
            )
        if not self.name:
            raise ValueError("Rule name must not be empty")
        if not self.expr:
            raise ValueError("Rule expr must not be empty")
        if self.for_duration < 0:
            raise ValueError("for_duration must be >= 0")
