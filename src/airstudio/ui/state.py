"""UI session state."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..engines.base import DeviceSpec, LLMEngine


@dataclass
class AppState:
    engine: LLMEngine | None = None
    current_model_key: str | None = None
    current_profile: str | None = None
    device: DeviceSpec | None = None
    engine_loaded: bool = False
    history: list[tuple[str, str]] = field(default_factory=list)
    last_metrics: dict | None = None

    def reset_history(self) -> None:
        self.history = []
