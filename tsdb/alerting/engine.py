"""Alerting engine: evaluates rules and manages alert lifecycle.

The AlertManager iterates over registered rules, evaluates each one's
expression, compares the result against the threshold, and drives each
alert through the state machine:

    condition becomes true   → PENDING  (started_at recorded)
    PENDING + for_duration elapsed      → FIRING  (fired_at recorded)
    condition becomes false  → RESOLVED (resolved_at recorded)
    RESOLVED + grace period elapsed     → removed from active set

Thread safety: all mutations to ``_active_alerts`` are protected by a lock
so that background evaluation and HTTP reads can coexist safely.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from tsdb.alerting.rule import AlertRule
from tsdb.alerting.state import Alert, AlertFingerprint, AlertState

if TYPE_CHECKING:
    from tsdb.ql.evaluator import QueryEvaluator

logger = logging.getLogger(__name__)

# How long a RESOLVED alert stays in the active set before being dropped.
_GRACE_PERIOD_SECONDS = 5.0


def _check_condition(value: float, condition: str, threshold: float) -> bool:
    """Return True when *value* satisfies *condition* against *threshold*."""
    if condition == "gt":
        return value > threshold
    if condition == "lt":
        return value < threshold
    if condition == "gte":
        return value >= threshold
    if condition == "lte":
        return value <= threshold
    if condition == "eq":
        return value == threshold
    if condition == "neq":
        return value != threshold
    raise ValueError(f"Unknown condition: {condition!r}")


def _scalar_from_results(results) -> float | None:
    """Collapse a list of QueryResult into a single scalar.

    For multi-series results the maximum of all latest sample values is used.
    Returns None when no samples are available.
    """
    if not results:
        return None

    candidates: list[float] = []
    for qr in results:
        if qr.samples:
            latest = max(qr.samples, key=lambda s: s.timestamp)
            candidates.append(latest.value)

    if not candidates:
        return None

    return max(candidates)


class AlertManager:
    """Evaluates alert rules and maintains the alert state machine.

    Parameters
    ----------
    query_evaluator:
        A :class:`~tsdb.ql.evaluator.QueryEvaluator` bound to the running
        engine.  The manager calls ``evaluate()`` on it for each rule.
    rules:
        Optional initial list of :class:`~tsdb.alerting.rule.AlertRule`
        objects.
    """

    def __init__(
        self,
        query_evaluator: "QueryEvaluator",
        rules: list[AlertRule] | None = None,
    ) -> None:
        self._evaluator = query_evaluator
        self._rules: list[AlertRule] = list(rules or [])
        self._active_alerts: dict[AlertFingerprint, Alert] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new rule.  Replaces any existing rule with the same name."""
        with self._lock:
            self._rules = [r for r in self._rules if r.name != rule.name]
            self._rules.append(rule)
        logger.info("Alert rule added: %s", rule.name)

    def remove_rule(self, name: str) -> None:
        """Remove the rule with the given name (no-op if it does not exist)."""
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.name != name]
            if len(self._rules) < before:
                logger.info("Alert rule removed: %s", name)

    def list_rules(self) -> list[AlertRule]:
        with self._lock:
            return list(self._rules)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, now: float | None = None) -> list[Alert]:
        """Evaluate every registered rule at time *now*.

        Returns the list of all currently active (non-INACTIVE) alerts after
        updating the state machine.
        """
        if now is None:
            now = time.time()

        with self._lock:
            rules = list(self._rules)

        for rule in rules:
            try:
                self._evaluate_rule(rule, now)
            except Exception:
                logger.exception("Error evaluating rule %r", rule.name)

        with self._lock:
            self._purge_stale(now)

        return self.get_alerts()

    def _evaluate_rule(self, rule: AlertRule, now: float) -> None:
        """Evaluate a single rule and update the state machine for its alerts."""
        # Use a 5-minute look-back window for instant evaluation.
        window = 300.0
        start = now - window
        end = now

        try:
            results = self._evaluator.evaluate(rule.expr, start, end, step=window)
        except Exception:
            logger.exception("Rule %r: expression evaluation failed", rule.name)
            return

        scalar = _scalar_from_results(results)
        if scalar is None:
            # No data: treat as condition-false for all currently pending/firing
            # alerts that belong to this rule.
            with self._lock:
                self._resolve_rule_alerts(rule.name, now, value=0.0)
            return

        # Build merged labels for this alert instance.
        merged_labels = dict(rule.labels)
        merged_labels["alertname"] = rule.name
        merged_labels["severity"] = rule.severity.value

        fp = AlertFingerprint.build(rule.name, merged_labels)
        condition_true = _check_condition(scalar, rule.condition, rule.threshold)

        with self._lock:
            existing = self._active_alerts.get(fp)

            if condition_true:
                if existing is None or existing.state == AlertState.RESOLVED:
                    # New alert: enter PENDING.
                    alert = Alert(
                        rule_name=rule.name,
                        labels=merged_labels,
                        state=AlertState.PENDING,
                        value=scalar,
                        started_at=now,
                    )
                    self._active_alerts[fp] = alert
                    logger.debug("Alert PENDING: %s (value=%.4f)", rule.name, scalar)

                elif existing.state == AlertState.PENDING:
                    existing.value = scalar
                    elapsed = now - (existing.started_at or now)
                    if elapsed >= rule.for_duration:
                        existing.state = AlertState.FIRING
                        existing.fired_at = now
                        logger.warning(
                            "Alert FIRING: %s (value=%.4f, threshold=%.4f)",
                            rule.name, scalar, rule.threshold,
                        )

                elif existing.state == AlertState.FIRING:
                    existing.value = scalar  # keep value current

            else:
                # Condition is false.
                if existing is not None and existing.state in (
                    AlertState.PENDING, AlertState.FIRING
                ):
                    existing.state = AlertState.RESOLVED
                    existing.resolved_at = now
                    existing.value = scalar
                    logger.info("Alert RESOLVED: %s", rule.name)

    def _resolve_rule_alerts(
        self, rule_name: str, now: float, value: float
    ) -> None:
        """Transition all PENDING/FIRING alerts for *rule_name* to RESOLVED."""
        for alert in self._active_alerts.values():
            if alert.rule_name == rule_name and alert.state in (
                AlertState.PENDING, AlertState.FIRING
            ):
                alert.state = AlertState.RESOLVED
                alert.resolved_at = now
                alert.value = value

    def _purge_stale(self, now: float) -> None:
        """Remove RESOLVED alerts that have passed the grace period."""
        to_delete = [
            fp
            for fp, alert in self._active_alerts.items()
            if alert.state == AlertState.RESOLVED
            and alert.resolved_at is not None
            and (now - alert.resolved_at) >= _GRACE_PERIOD_SECONDS
        ]
        for fp in to_delete:
            del self._active_alerts[fp]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_alerts(self, state: AlertState | None = None) -> list[Alert]:
        """Return active alerts, optionally filtered to a specific state.

        Stale RESOLVED alerts are purged during ``evaluate_all``; this method
        only reads the current snapshot.
        """
        with self._lock:
            alerts = list(self._active_alerts.values())

        if state is not None:
            alerts = [a for a in alerts if a.state == state]
        return alerts

    def get_firing(self) -> list[Alert]:
        """Return only FIRING alerts."""
        return self.get_alerts(state=AlertState.FIRING)
