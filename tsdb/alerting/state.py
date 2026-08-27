"""Alert state machine types.

Each distinct combination of rule + label-set produces an independent Alert
instance.  The Alert moves through states: INACTIVE → PENDING → FIRING and
then → RESOLVED before being removed from the active set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AlertState(str, Enum):
    INACTIVE = "inactive"
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"


class AlertFingerprint(str):
    """A stable, hashable identifier for an alert instance.

    Built from the rule name and the alert's label set so that two alerts
    with the same rule and labels always map to the same fingerprint, even
    across evaluation cycles.
    """

    @classmethod
    def build(cls, rule_name: str, labels: dict[str, str]) -> "AlertFingerprint":
        parts = [f"rule={rule_name}"]
        for k, v in sorted(labels.items()):
            parts.append(f"{k}={v}")
        return cls(",".join(parts))


@dataclass
class Alert:
    """One active (or recently resolved) alert instance.

    Attributes
    ----------
    rule_name:
        Name of the rule that produced this alert.
    labels:
        Combined labels: rule labels merged with any series labels from the
        query result.
    state:
        Current lifecycle state.
    value:
        The metric value that most recently caused the condition to be true
        (or the last observed value when the alert is RESOLVED).
    started_at:
        Unix timestamp when the condition first became true (entry into
        PENDING).
    fired_at:
        Unix timestamp when the alert transitioned to FIRING.
    resolved_at:
        Unix timestamp when the alert transitioned to RESOLVED.
    """

    rule_name: str
    labels: dict[str, str]
    state: AlertState
    value: float
    started_at: float | None = None
    fired_at: float | None = None
    resolved_at: float | None = None

    def fingerprint(self) -> AlertFingerprint:
        return AlertFingerprint.build(self.rule_name, self.labels)
