"""RAM monitor for peak RSS."""
from __future__ import annotations

import threading
import time

import psutil


class RamMonitor:
    def __init__(self, interval_ms: int) -> None:
        self._interval = interval_ms / 1000.0
        self._peak = 0
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak / (1024 * 1024)

    def _run(self) -> None:
        proc = psutil.Process()
        while self._running.is_set():
            rss = proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            time.sleep(self._interval)
