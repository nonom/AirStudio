"""Instrumentation wrapper for generate() calls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ram_monitor import RamMonitor
from .vram_monitor import VramMonitor
from ..engines.base import GenerationOutput


@dataclass
class MeasuredOutput:
    output: GenerationOutput
    tokens_per_s: float
    ram_peak_mb: float
    vram_peak_mb: float | None


class Instrumentation:
    def __init__(self, sampling_interval_ms: int, gpu_index: int | None) -> None:
        self._interval = sampling_interval_ms
        self._gpu_index = gpu_index

    def measure_generate(self, fn: Callable[[], GenerationOutput]) -> MeasuredOutput:
        ram = RamMonitor(self._interval)
        vram = VramMonitor(self._interval, self._gpu_index)
        ram.start()
        vram.start()
        output = fn()
        ram_peak = ram.stop()
        vram_peak = vram.stop()
        tokens_per_s = 0.0
        if output.decode_time_s > 0:
            tokens_per_s = output.generated_tokens / output.decode_time_s
        return MeasuredOutput(
            output=output,
            tokens_per_s=tokens_per_s,
            ram_peak_mb=ram_peak,
            vram_peak_mb=vram_peak,
        )
