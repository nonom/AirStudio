"""Model registry helpers."""
from __future__ import annotations

from .config import ModelSpec


class ModelRegistry:
    def __init__(self, models: list[ModelSpec]):
        self._models = models

    def list(self) -> list[ModelSpec]:
        return list(self._models)

    def get(self, key: str) -> ModelSpec:
        for model in self._models:
            if model.key == key:
                return model
        raise KeyError(f"Model not found: {key}")
