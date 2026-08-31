"""Recording rules: pre-compute expensive PromQL queries on a schedule.

A recording rule evaluates a PromQL expression at a fixed interval and writes
the result back into the engine as a new time series.  This is the same
concept as Prometheus recording rules — it lets you materialise costly
aggregations so dashboards and alert rules can query the pre-computed series
cheaply instead of re-running the full expression every time.

The RecordingRuleManager maintains one daemon thread per rule group.  Each
thread sleeps for the group's interval, wakes up, evaluates every rule in the
group, and writes the results back.  Threads are torn down cleanly when
``stop()`` is called.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tsdb.engine import Engine
    from tsdb.ql.evaluator import QueryEvaluator

logger = logging.getLogger(__name__)

# How far back (seconds) to look when running an instant evaluation.
_LOOKBACK_WINDOW = 300.0


@dataclass
class RecordingRule:
    """A single recording rule.

    Attributes
    ----------
    name:
        The metric name that results will be written as.
    expr:
        PromQL expression to evaluate.
    interval:
        How often (in seconds) the rule is evaluated.  Defaults to 60 s.
        Usually inherited from the containing group and not set explicitly.
    labels:
        Extra labels merged into every series the rule produces.  These
        take precedence over labels already on the series.
    """

    name: str
    expr: str
    interval: float = 60.0
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RecordingRule.name must not be empty")
        if not self.expr:
            raise ValueError("RecordingRule.expr must not be empty")
        if self.interval <= 0:
            raise ValueError("RecordingRule.interval must be > 0")


@dataclass
class RecordingRuleGroup:
    """A named collection of recording rules that share an evaluation interval.

    Attributes
    ----------
    name:
        Unique group identifier.
    interval:
        Default evaluation interval (seconds) for rules in this group.
    rules:
        Ordered list of :class:`RecordingRule` objects belonging to this group.
    """

    name: str
    interval: float = 60.0
    rules: list[RecordingRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RecordingRuleGroup.name must not be empty")
        if self.interval <= 0:
            raise ValueError("RecordingRuleGroup.interval must be > 0")


class RecordingRuleManager:
    """Manages recording rule groups and their background evaluation threads.

    One daemon thread is spawned per group when ``start()`` is called.  Each
    thread evaluates all rules in the group at the group's interval.

    Parameters
    ----------
    engine:
        The running storage engine.  Results are written back via
        ``engine.write()``.
    evaluator:
        A :class:`~tsdb.ql.evaluator.QueryEvaluator` bound to the same engine.
    """

    def __init__(self, engine: "Engine", evaluator: "QueryEvaluator") -> None:
        self._engine = engine
        self._evaluator = evaluator
        self._groups: list[RecordingRuleGroup] = []
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Group management
    # ------------------------------------------------------------------

    def load_groups(self, groups: list[RecordingRuleGroup]) -> None:
        """Replace the current set of rule groups.

        If the manager is already running, the old threads are stopped first
        and new ones are started for the new groups.
        """
        was_running = any(t.is_alive() for t in self._threads)

        if was_running:
            self._stop_threads()

        with self._lock:
            self._groups = list(groups)
            logger.info(
                "RecordingRuleManager: loaded %d group(s) with %d rule(s) total",
                len(self._groups),
                sum(len(g.rules) for g in self._groups),
            )

        if was_running:
            self._start_threads()

    def list_groups(self) -> list[RecordingRuleGroup]:
        with self._lock:
            return list(self._groups)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_group(self, group: RecordingRuleGroup) -> None:
        """Evaluate all rules in *group* immediately."""
        for rule in group.rules:
            try:
                self.evaluate_rule(rule)
            except Exception:
                logger.exception(
                    "RecordingRule %r in group %r raised an error during evaluation",
                    rule.name, group.name,
                )

    def evaluate_rule(self, rule: RecordingRule) -> None:
        """Evaluate *rule* at the current time and write results back.

        Steps:
        1. Run the PromQL expression over a recent look-back window.
        2. For each resulting series, write a new sample using the rule's
           ``name`` as the metric name, merging rule labels on top of
           the series' existing labels.
        """
        now = time.time()
        start = now - _LOOKBACK_WINDOW
        end = now

        try:
            results = self._evaluator.evaluate(rule.expr, start, end, step=_LOOKBACK_WINDOW)
        except Exception:
            logger.exception("RecordingRule %r: expression evaluation failed", rule.name)
            return

        if not results:
            logger.debug("RecordingRule %r: no results, nothing written", rule.name)
            return

        written = 0
        for qr in results:
            if not qr.samples:
                continue

            # Take the most recent sample as the instant value.
            latest = max(qr.samples, key=lambda s: s.timestamp)

            # Merge labels: series labels first, then rule labels override.
            merged = dict(qr.key.labels)
            merged.update(rule.labels)

            try:
                self._engine.write(rule.name, merged, now, latest.value)
                written += 1
            except Exception:
                logger.exception(
                    "RecordingRule %r: failed to write result for labels %r",
                    rule.name, merged,
                )

        logger.debug("RecordingRule %r: wrote %d series at t=%.3f", rule.name, written, now)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn a background daemon thread for each rule group."""
        self._stop_event.clear()
        self._start_threads()

    def stop(self) -> None:
        """Signal all group threads to stop and wait for them to finish."""
        self._stop_threads()

    def _start_threads(self) -> None:
        with self._lock:
            groups = list(self._groups)

        for group in groups:
            t = threading.Thread(
                target=self._run_group,
                args=(group,),
                name=f"tsdb-recording-{group.name}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)
            logger.info(
                "RecordingRuleGroup %r: background thread started (interval=%.1fs, %d rule(s))",
                group.name, group.interval, len(group.rules),
            )

    def _stop_threads(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=10)
        self._threads.clear()
        self._stop_event.clear()
        logger.info("RecordingRuleManager: all group threads stopped")

    def _run_group(self, group: RecordingRuleGroup) -> None:
        """Thread target: evaluate the group on its interval until stopped."""
        logger.debug("RecordingRuleGroup %r: thread started", group.name)
        while not self._stop_event.is_set():
            self.evaluate_group(group)
            # Sleep in short bursts so we react to stop quickly.
            self._stop_event.wait(timeout=group.interval)
        logger.debug("RecordingRuleGroup %r: thread exiting", group.name)
