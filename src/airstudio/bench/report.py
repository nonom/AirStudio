"""Benchmark reporting utilities."""
from __future__ import annotations

from datetime import datetime
from typing import Any


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    error_count = 0

    for row in rows:
        if row.get("error"):
            error_count += 1
            continue
        key = (row["model_key"], row["profile"], int(row["context_tokens"]))
        group = groups.setdefault(
            key,
            {
                "model_key": row["model_key"],
                "profile": row["profile"],
                "context_tokens": int(row["context_tokens"]),
                "count": 0,
                "tokens_per_s": 0.0,
                "decode_time_s": 0.0,
                "prompt_tokens": 0.0,
                "generated_tokens": 0.0,
                "ram_peak_mb": 0.0,
                "vram_peak_mb": 0.0,
                "vram_count": 0,
            },
        )
        group["count"] += 1
        group["tokens_per_s"] += float(row.get("tokens_per_s") or 0)
        group["decode_time_s"] += float(row.get("decode_time_s") or 0)
        group["prompt_tokens"] += float(row.get("prompt_tokens") or 0)
        group["generated_tokens"] += float(row.get("generated_tokens") or 0)
        group["ram_peak_mb"] += float(row.get("ram_peak_mb") or 0)
        vram = row.get("vram_peak_mb")
        if vram is not None:
            group["vram_peak_mb"] += float(vram)
            group["vram_count"] += 1

    group_list: list[dict[str, Any]] = []
    for group in groups.values():
        count = max(1, int(group["count"]))
        vram_count = max(1, int(group["vram_count"]))
        group_list.append(
            {
                "model_key": group["model_key"],
                "profile": group["profile"],
                "context_tokens": group["context_tokens"],
                "count": group["count"],
                "tokens_per_s": group["tokens_per_s"] / count,
                "decode_time_s": group["decode_time_s"] / count,
                "prompt_tokens": group["prompt_tokens"] / count,
                "generated_tokens": group["generated_tokens"] / count,
                "ram_peak_mb": group["ram_peak_mb"] / count,
                "vram_peak_mb": group["vram_peak_mb"] / vram_count,
            }
        )

    return {
        "generated_at": datetime.now().isoformat(),
        "num_runs": len(rows),
        "num_errors": error_count,
        "groups": group_list,
    }
