"""Background task runner.

Runs periodic maintenance functions (retention, compaction, checkpointing)
on a daemon thread so the main request-serving path stays unblocked.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class BackgroundTask:
    name: str
    interval_seconds: float
    fn: Callable
    last_run: float = field(default=0.0)


class BackgroundWorker:
    def __init__(self) -> None:
        self._tasks: list[BackgroundTask] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def register(self, name: str, interval_seconds: float, fn: Callable) -> None:
        """Add a task that will be called every interval_seconds."""
        self._tasks.append(BackgroundTask(name=name, interval_seconds=interval_seconds, fn=fn))

    def start(self) -> None:
        """Start the background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="tsdb-background", daemon=True)
        self._thread.start()
        logger.info("BackgroundWorker started with %d task(s)", len(self._tasks))

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("BackgroundWorker stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            for task in self._tasks:
                if now - task.last_run >= task.interval_seconds:
                    try:
                        task.fn()
                    except Exception:
                        logger.exception("BackgroundTask %r raised an exception", task.name)
                    task.last_run = time.time()
            # Sleep in short intervals so stop_event is noticed promptly.
            self._stop_event.wait(timeout=1.0)
