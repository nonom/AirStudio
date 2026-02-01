"""Engine protocol and dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class DeviceSpec:
    kind: Literal["cuda", "cpu"]
    gpu_index: int | None


@dataclass
class GenerationSpec:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    use_cache: bool
    max_context: int


@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int
    generated_tokens: int
    decode_time_s: float


class LLMEngine(Protocol):
    def load(self, model_path: str, compression: str | None, layer_cache_dir: str, device: DeviceSpec) -> None:
        ...

    def unload(self) -> None:
        ...

    def tokenizer(self):
        ...

    def generate(
        self, messages: list[dict], gen: GenerationSpec, family_hint: str | None
    ) -> GenerationOutput:
        ...
