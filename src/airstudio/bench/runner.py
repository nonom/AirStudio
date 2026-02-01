"""Benchmark runner for AirStudio."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import csv
import os
from typing import Any, Iterable

import torch

from ..config import AppConfig, BenchConfig, GenerationDefaults, ProfileSpec
from ..engines.airllm_engine import AirLLMEngine
from ..engines.base import DeviceSpec, GenerationSpec
from ..metrics.instrumentation import Instrumentation
from ..prompts import build_messages_for_context
from ..registry import ModelRegistry
from .report import summarize_results


RESULT_FIELDS = [
    "model_key",
    "profile",
    "context_tokens",
    "repeat_idx",
    "prompt_tokens",
    "generated_tokens",
    "decode_time_s",
    "tokens_per_s",
    "ram_peak_mb",
    "vram_peak_mb",
    "error",
]


@dataclass
class BenchResult:
    model_key: str
    profile: str
    context_tokens: int
    repeat_idx: int
    prompt_tokens: int | None
    generated_tokens: int | None
    decode_time_s: float | None
    tokens_per_s: float | None
    ram_peak_mb: float | None
    vram_peak_mb: float | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "profile": self.profile,
            "context_tokens": self.context_tokens,
            "repeat_idx": self.repeat_idx,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "decode_time_s": self.decode_time_s,
            "tokens_per_s": self.tokens_per_s,
            "ram_peak_mb": self.ram_peak_mb,
            "vram_peak_mb": self.vram_peak_mb,
            "error": self.error,
        }


def _build_device(cfg: AppConfig) -> DeviceSpec:
    if torch.cuda.is_available() and cfg.gpu_index is not None and cfg.gpu_index >= 0:
        return DeviceSpec(kind="cuda", gpu_index=cfg.gpu_index)
    return DeviceSpec(kind="cpu", gpu_index=None)


def _resolve_cache_dir(base_dir: str, model_key: str) -> str:
    if not base_dir:
        return base_dir
    if "{model_key}" in base_dir:
        return base_dir.replace("{model_key}", model_key)
    base = os.path.basename(base_dir.rstrip("/\\"))
    if base != model_key:
        return os.path.join(base_dir, model_key)
    return base_dir


def _timestamp_dir(root: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _write_csv(path: str, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_benchmark(
    registry: ModelRegistry,
    profiles: dict[str, ProfileSpec],
    app_cfg: AppConfig,
    gen_defaults: GenerationDefaults,
    bench_cfg: BenchConfig,
    model_keys: list[str],
    profile_keys: list[str],
    contexts: list[int],
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    results: list[BenchResult] = []
    output_dir = _timestamp_dir(bench_cfg.output_dir)

    engine = AirLLMEngine()
    loaded_model: str | None = None
    loaded_profile: str | None = None

    device = _build_device(app_cfg)

    for model_key in model_keys:
        model = registry.get(model_key)
        if not os.path.exists(model.local_path):
            for profile_key in profile_keys:
                for context in contexts:
                    results.append(
                        BenchResult(
                            model_key=model_key,
                            profile=profile_key,
                            context_tokens=int(context),
                            repeat_idx=0,
                            prompt_tokens=None,
                            generated_tokens=None,
                            decode_time_s=None,
                            tokens_per_s=None,
                            ram_peak_mb=None,
                            vram_peak_mb=None,
                            error=f"Model path not found: {model.local_path}",
                        )
                    )
            continue

        for profile_key in profile_keys:
            profile = profiles.get(profile_key)
            if profile is None:
                for context in contexts:
                    results.append(
                        BenchResult(
                            model_key=model_key,
                            profile=profile_key,
                            context_tokens=int(context),
                            repeat_idx=0,
                            prompt_tokens=None,
                            generated_tokens=None,
                            decode_time_s=None,
                            tokens_per_s=None,
                            ram_peak_mb=None,
                            vram_peak_mb=None,
                            error=f"Profile not found: {profile_key}",
                        )
                    )
                continue

            if loaded_model != model_key or loaded_profile != profile_key:
                if loaded_model is not None or loaded_profile is not None:
                    engine.unload()
                cache_dir = _resolve_cache_dir(profile.layer_cache_dir, model.key)
                engine.load(
                    model_path=model.local_path,
                    compression=profile.compression,
                    layer_cache_dir=cache_dir,
                    device=device,
                )
                loaded_model = model_key
                loaded_profile = profile_key

            tokenizer = engine.tokenizer()

            for context in contexts:
                try:
                    messages = build_messages_for_context(tokenizer, model.family_hint, int(context))
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        BenchResult(
                            model_key=model_key,
                            profile=profile_key,
                            context_tokens=int(context),
                            repeat_idx=0,
                            prompt_tokens=None,
                            generated_tokens=None,
                            decode_time_s=None,
                            tokens_per_s=None,
                            ram_peak_mb=None,
                            vram_peak_mb=None,
                            error=str(exc),
                        )
                    )
                    continue

                if bench_cfg.warmup:
                    warm_gen = GenerationSpec(
                        max_new_tokens=int(bench_cfg.warmup_new_tokens),
                        temperature=float(gen_defaults.temperature),
                        top_p=float(gen_defaults.top_p),
                        do_sample=bool(gen_defaults.do_sample),
                        use_cache=bool(gen_defaults.use_cache),
                        max_context=int(context),
                    )
                    try:
                        engine.generate(messages, warm_gen, model.family_hint)
                    except Exception:
                        pass

                for repeat_idx in range(int(bench_cfg.repeats)):
                    gen = GenerationSpec(
                        max_new_tokens=int(bench_cfg.measure_new_tokens),
                        temperature=float(gen_defaults.temperature),
                        top_p=float(gen_defaults.top_p),
                        do_sample=bool(gen_defaults.do_sample),
                        use_cache=bool(gen_defaults.use_cache),
                        max_context=int(context),
                    )
                    instr = Instrumentation(app_cfg.sampling_interval_ms, app_cfg.gpu_index)
                    try:
                        measured = instr.measure_generate(
                            lambda: engine.generate(messages, gen, model.family_hint)
                        )
                        results.append(
                            BenchResult(
                                model_key=model_key,
                                profile=profile_key,
                                context_tokens=int(context),
                                repeat_idx=int(repeat_idx),
                                prompt_tokens=int(measured.output.prompt_tokens),
                                generated_tokens=int(measured.output.generated_tokens),
                                decode_time_s=float(measured.output.decode_time_s),
                                tokens_per_s=float(measured.tokens_per_s),
                                ram_peak_mb=float(measured.ram_peak_mb),
                                vram_peak_mb=(
                                    float(measured.vram_peak_mb)
                                    if measured.vram_peak_mb is not None
                                    else None
                                ),
                                error=None,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        results.append(
                            BenchResult(
                                model_key=model_key,
                                profile=profile_key,
                                context_tokens=int(context),
                                repeat_idx=int(repeat_idx),
                                prompt_tokens=None,
                                generated_tokens=None,
                                decode_time_s=None,
                                tokens_per_s=None,
                                ram_peak_mb=None,
                                vram_peak_mb=None,
                                error=str(exc),
                            )
                        )

    rows = [r.to_dict() for r in results]
    _write_csv(os.path.join(output_dir, "results.csv"), rows)
    summary = summarize_results(rows)
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        import json

        json.dump(summary, handle, indent=2)

    return rows, output_dir, summary
