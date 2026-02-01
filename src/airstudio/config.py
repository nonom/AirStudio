"""Configuration loading and dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class AppConfig:
    title: str = "AirStudio"
    host: str = "127.0.0.1"
    port: int = 7860
    concurrency_limit: int = 1
    sampling_interval_ms: int = 50
    offline_mode: bool = True
    gpu_index: int | None = 0


@dataclass
class GenerationDefaults:
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    use_cache: bool = True
    max_context: int = 4096


@dataclass
class ProfileSpec:
    key: str
    compression: str | None
    layer_cache_dir: str


@dataclass
class ModelSpec:
    key: str
    display_name: str
    local_path: str
    family_hint: str | None


@dataclass
class BenchConfig:
    enabled: bool = True
    contexts: list[int] = field(default_factory=lambda: [512, 4096])
    repeats: int = 3
    warmup: bool = True
    warmup_new_tokens: int = 32
    measure_new_tokens: int = 128
    output_dir: str = "runs"


@dataclass
class RootConfig:
    app: AppConfig
    generation_defaults: GenerationDefaults
    profiles: dict[str, ProfileSpec]
    models: list[ModelSpec]
    bench: BenchConfig


def _get(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get(key, default) if isinstance(data, dict) else default


def load_config(path: str) -> RootConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    app_raw = _get(raw, "app", {})
    gen_raw = _get(raw, "generation_defaults", {})
    profiles_raw = _get(raw, "profiles", {})
    models_raw = _get(raw, "models", [])
    bench_raw = _get(raw, "bench", {})

    app = AppConfig(
        title=_get(app_raw, "title", AppConfig.title),
        host=_get(app_raw, "host", AppConfig.host),
        port=int(_get(app_raw, "port", AppConfig.port)),
        concurrency_limit=int(_get(app_raw, "concurrency_limit", AppConfig.concurrency_limit)),
        sampling_interval_ms=int(_get(app_raw, "sampling_interval_ms", AppConfig.sampling_interval_ms)),
        offline_mode=bool(_get(app_raw, "offline_mode", AppConfig.offline_mode)),
        gpu_index=_get(app_raw, "gpu_index", AppConfig.gpu_index),
    )

    gen = GenerationDefaults(
        max_new_tokens=int(_get(gen_raw, "max_new_tokens", GenerationDefaults.max_new_tokens)),
        temperature=float(_get(gen_raw, "temperature", GenerationDefaults.temperature)),
        top_p=float(_get(gen_raw, "top_p", GenerationDefaults.top_p)),
        do_sample=bool(_get(gen_raw, "do_sample", GenerationDefaults.do_sample)),
        use_cache=bool(_get(gen_raw, "use_cache", GenerationDefaults.use_cache)),
        max_context=int(_get(gen_raw, "max_context", GenerationDefaults.max_context)),
    )

    profiles: dict[str, ProfileSpec] = {}
    if isinstance(profiles_raw, dict):
        for key, value in profiles_raw.items():
            profiles[key] = ProfileSpec(
                key=key,
                compression=_get(value, "compression", None),
                layer_cache_dir=_get(value, "layer_cache_dir", "./cache/airllm_layers"),
            )

    models: list[ModelSpec] = []
    if isinstance(models_raw, list):
        for item in models_raw:
            models.append(
                ModelSpec(
                    key=_get(item, "key", ""),
                    display_name=_get(item, "display_name", ""),
                    local_path=_get(item, "local_path", ""),
                    family_hint=_get(item, "family_hint", None),
                )
            )

    bench = BenchConfig(
        enabled=bool(_get(bench_raw, "enabled", BenchConfig.enabled)),
        contexts=_get(bench_raw, "contexts", BenchConfig().contexts),
        repeats=int(_get(bench_raw, "repeats", BenchConfig.repeats)),
        warmup=bool(_get(bench_raw, "warmup", BenchConfig.warmup)),
        warmup_new_tokens=int(_get(bench_raw, "warmup_new_tokens", BenchConfig.warmup_new_tokens)),
        measure_new_tokens=int(_get(bench_raw, "measure_new_tokens", BenchConfig.measure_new_tokens)),
        output_dir=_get(bench_raw, "output_dir", BenchConfig.output_dir),
    )

    return RootConfig(
        app=app,
        generation_defaults=gen,
        profiles=profiles,
        models=models,
        bench=bench,
    )



