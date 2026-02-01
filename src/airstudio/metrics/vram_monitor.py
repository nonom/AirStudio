"""VRAM monitor using NVML with fallback."""
from __future__ import annotations

import threading
import time

try:
    import pynvml  # provided by nvidia-ml-py
except Exception:  # pragma: no cover
    pynvml = None


class VramMonitor:
    def __init__(self, interval_ms: int, gpu_index: int | None) -> None:
        self._interval = interval_ms / 1000.0
        self._gpu_index = gpu_index if gpu_index is not None else 0
        self._peak = 0
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = pynvml is not None

    def start(self) -> None:
        if not self._enabled:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        except Exception:
            self._enabled = False
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if not self._enabled:
            return None
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return self._peak / (1024 * 1024)

    def _run(self) -> None:
        while self._running.is_set():
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used = info.used
                if used > self._peak:
                    self._peak = used
            except Exception:
                pass
            time.sleep(self._interval)
