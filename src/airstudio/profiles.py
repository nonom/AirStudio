"""Profile presets."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProfileSpec:
    key: str
    compression: str | None
    layer_cache_dir: str
