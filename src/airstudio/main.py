"""AirStudio UI entrypoint."""
from __future__ import annotations

import argparse
import os
from typing import Any

import gradio as gr
import torch

from .config import AppConfig, BenchConfig, GenerationDefaults, RootConfig, load_config
from .bench.runner import run_benchmark, RESULT_FIELDS
from .engines.airllm_engine import AirLLMEngine
from .engines.base import DeviceSpec, GenerationSpec
from .metrics.instrumentation import Instrumentation
from .prompts import build_chat_messages
from .registry import ModelRegistry
from .ui.state import AppState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirStudio UI")
    parser.add_argument("--config", default="configs/airstudio.yaml")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--title")
    parser.add_argument("--concurrency-limit", type=int)
    parser.add_argument("--sampling-interval-ms", type=int)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def load_root_config(path: str) -> RootConfig:
    if not os.path.exists(path):
        return RootConfig(
            app=AppConfig(),
            generation_defaults=GenerationDefaults(),
            profiles={},
            models=[],
            bench=BenchConfig(),
        )
    return load_config(path)


def apply_overrides(cfg: RootConfig, args: argparse.Namespace) -> RootConfig:
    if args.title:
        cfg.app.title = args.title
    if args.host:
        cfg.app.host = args.host
    if args.port is not None:
        cfg.app.port = args.port
    if args.concurrency_limit is not None:
        cfg.app.concurrency_limit = args.concurrency_limit
    if args.sampling_interval_ms is not None:
        cfg.app.sampling_interval_ms = args.sampling_interval_ms
    if args.gpu_index is not None:
        cfg.app.gpu_index = args.gpu_index
    if args.offline:
        cfg.app.offline_mode = True
    return cfg


def ensure_offline(cfg: AppConfig) -> None:
    if cfg.offline_mode:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _build_device(cfg: AppConfig) -> DeviceSpec:
    if torch.cuda.is_available() and cfg.gpu_index is not None and cfg.gpu_index >= 0:
        return DeviceSpec(kind="cuda", gpu_index=cfg.gpu_index)
    return DeviceSpec(kind="cpu", gpu_index=None)


def _metrics_markdown(metrics: dict[str, Any]) -> str:
    if metrics.get("error"):
        return f"**error:** {metrics['error']}"
    vram = metrics.get("vram_peak_mb")
    vram_str = f"{vram:.2f}" if isinstance(vram, (int, float)) else "n/a"
    return (
        f"**tokens/s:** {metrics.get('tokens_per_s', 0):.2f}\n"
        f"**prompt_tokens:** {metrics.get('prompt_tokens', 0)}\n"
        f"**generated_tokens:** {metrics.get('generated_tokens', 0)}\n"
        f"**ram_peak_mb:** {metrics.get('ram_peak_mb', 0):.2f}\n"
        f"**vram_peak_mb:** {vram_str}\n"
        f"**decode_time_s:** {metrics.get('decode_time_s', 0):.4f}"
    )


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "content" in content:
            return _extract_text(content.get("content"))
        if "text" in content:
            return str(content.get("text"))
    if isinstance(content, list):
        return "".join(_extract_text(item) for item in content)
    return str(content)


def _history_to_pairs(history: list[Any]) -> list[tuple[str, str]]:
    if not history:
        return []
    first = history[0]
    if isinstance(first, (tuple, list)) and len(first) == 2:
        return [
            (str(item[0]), str(item[1]))
            for item in history
            if isinstance(item, (tuple, list)) and len(item) == 2
        ]
    pairs: list[tuple[str, str]] = []
    last_user: str | None = None
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = _extract_text(item.get("content"))
        if role == "user":
            last_user = content
        elif role == "assistant":
            if last_user is None:
                pairs.append(("", content))
            else:
                pairs.append((last_user, content))
                last_user = None
    return pairs


def _pairs_to_messages(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user, assistant in pairs:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


def build_app(cfg: RootConfig, args: argparse.Namespace) -> gr.Blocks:
    registry = ModelRegistry(cfg.models)
    profiles = cfg.profiles
    state = AppState(engine=AirLLMEngine())

    model_choices = [(m.display_name, m.key) for m in registry.list()]
    profile_choices = [
        (
            f"{key} ({profile.compression or 'none'})",
            key,
        )
        for key, profile in profiles.items()
    ]

    default_model = model_choices[0][1] if model_choices else None
    default_profile = profile_choices[0][1] if profile_choices else None

    with gr.Blocks(title=cfg.app.title) as demo:
        gr.Markdown(f"# {cfg.app.title}")

        with gr.Tab("Chat"):
            with gr.Row():
                model_dd = gr.Dropdown(
                    label="Model",
                    choices=model_choices,
                    value=default_model,
                )
                profile_dd = gr.Dropdown(
                    label="Profile",
                    choices=profile_choices,
                    value=default_profile,
                )

            with gr.Row():
                max_context = gr.Slider(
                    minimum=256,
                    maximum=8192,
                    step=256,
                    value=cfg.generation_defaults.max_context,
                    label="Max context",
                )
                max_new = gr.Slider(
                    minimum=1,
                    maximum=2048,
                    step=1,
                    value=cfg.generation_defaults.max_new_tokens,
                    label="Max new tokens",
                )

            with gr.Accordion("Advanced", open=False):
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=cfg.generation_defaults.temperature,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=cfg.generation_defaults.top_p,
                    label="Top-p",
                )

                do_sample = gr.Checkbox(
                    value=cfg.generation_defaults.do_sample,
                    label="Sampling (do_sample)"
                )
            chatbot = gr.Chatbot(label="Chat")
            user_input = gr.Textbox(label="Message", placeholder="Type a message...")
            send_btn = gr.Button("Send")
            metrics_md = gr.Markdown("No metrics yet.")

            def _handle_chat(
                message: str,
                history: list[Any],
                model_key: str,
                profile_key: str,
                max_context_val: int,
                max_new_val: int,
                temperature_val: float,
                top_p_val: float,
                do_sample_val: bool,
            ):
                pairs = _history_to_pairs(history)
                if not message:
                    return _pairs_to_messages(pairs), metrics_md.value, ""
                try:
                    device = _build_device(cfg.app)
                    if state.engine is None:
                        state.engine = AirLLMEngine()

                    if (
                        not state.engine_loaded
                        or state.current_model_key != model_key
                        or state.current_profile != profile_key
                    ):
                        if model_key is None:
                            raise RuntimeError("No model selected")
                        if profile_key is None:
                            raise RuntimeError("No profile selected")
                        model = registry.get(model_key)
                        profile = profiles.get(profile_key)
                        if profile is None:
                            raise RuntimeError(f"Profile not found: {profile_key}")
                        if not os.path.exists(model.local_path):
                            raise RuntimeError(f"Model path not found: {model.local_path}")
                        cache_dir = profile.layer_cache_dir
                        if cache_dir:
                            if "{model_key}" in cache_dir:
                                cache_dir = cache_dir.replace("{model_key}", model.key)
                            else:
                                base = os.path.basename(cache_dir.rstrip("/\\"))
                                if base != model.key:
                                    cache_dir = os.path.join(cache_dir, model.key)
                        state.engine.load(
                            model_path=model.local_path,
                            compression=profile.compression,
                            layer_cache_dir=cache_dir,
                            device=device,
                        )
                        state.engine_loaded = True
                        state.current_model_key = model_key
                        state.current_profile = profile_key
                        state.device = device

                    messages = build_chat_messages(pairs, message)
                    gen = GenerationSpec(
                        max_new_tokens=int(max_new_val),
                        temperature=float(temperature_val),
                        top_p=float(top_p_val),
                        do_sample=bool(do_sample_val),
                        use_cache=cfg.generation_defaults.use_cache,
                        max_context=int(max_context_val),
                    )
                    instr = Instrumentation(cfg.app.sampling_interval_ms, cfg.app.gpu_index)
                    measured = instr.measure_generate(
                        lambda: state.engine.generate(messages, gen, None)
                    )
                    reply = measured.output.text
                    pairs = pairs + [(message, reply)]
                    metrics = {
                        "tokens_per_s": measured.tokens_per_s,
                        "prompt_tokens": measured.output.prompt_tokens,
                        "generated_tokens": measured.output.generated_tokens,
                        "ram_peak_mb": measured.ram_peak_mb,
                        "vram_peak_mb": measured.vram_peak_mb,
                        "decode_time_s": measured.output.decode_time_s,
                    }
                    return _pairs_to_messages(pairs), _metrics_markdown(metrics), ""
                except Exception as exc:  # noqa: BLE001
                    pairs = pairs + [(message, f"[error] {exc}")]
                    metrics = {"error": str(exc)}
                    return _pairs_to_messages(pairs), _metrics_markdown(metrics), ""

            send_btn.click(
                _handle_chat,
                inputs=[
                    user_input,
                    chatbot,
                    model_dd,
                    profile_dd,
                    max_context,
                    max_new,
                    temperature,
                    top_p,
                    do_sample,
                ],
                outputs=[chatbot, metrics_md, user_input],
            )
            user_input.submit(
                _handle_chat,
                inputs=[
                    user_input,
                    chatbot,
                    model_dd,
                    profile_dd,
                    max_context,
                    max_new,
                    temperature,
                    top_p,
                    do_sample,
                ],
                outputs=[chatbot, metrics_md, user_input],
            )

        with gr.Tab("Benchmark"):
            with gr.Row():
                bench_models = gr.CheckboxGroup(
                    choices=model_choices,
                    value=[default_model] if default_model else [],
                    label="Models",
                )
                bench_profiles = gr.CheckboxGroup(
                    choices=profile_choices,
                    value=[default_profile] if default_profile else [],
                    label="Profiles",
                )

            bench_contexts = gr.CheckboxGroup(
                choices=cfg.bench.contexts,
                value=cfg.bench.contexts,
                label="Contexts",
            )

            with gr.Row():
                bench_repeats = gr.Slider(
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=cfg.bench.repeats,
                    label="Repeats",
                )
                bench_warmup = gr.Checkbox(
                    value=cfg.bench.warmup,
                    label="Warmup",
                )

            with gr.Row():
                bench_warmup_tokens = gr.Slider(
                    minimum=1,
                    maximum=256,
                    step=1,
                    value=cfg.bench.warmup_new_tokens,
                    label="Warmup new tokens",
                )
                bench_measure_tokens = gr.Slider(
                    minimum=1,
                    maximum=512,
                    step=1,
                    value=cfg.bench.measure_new_tokens,
                    label="Measure new tokens",
                )

            bench_run = gr.Button("Run benchmark")
            bench_status = gr.Markdown("")
            bench_results = gr.Dataframe(headers=RESULT_FIELDS, label="Results", interactive=False)
            bench_output = gr.Textbox(label="Output dir", interactive=False)

            def _run_benchmark_ui(
                models_sel: list[str],
                profiles_sel: list[str],
                contexts_sel: list[int],
                repeats_val: int,
                warmup_val: bool,
                warmup_tokens_val: int,
                measure_tokens_val: int,
            ):
                if not models_sel or not profiles_sel:
                    return [], "", "**error:** select at least one model and profile"
                if not contexts_sel:
                    return [], "", "**error:** select at least one context"
                contexts = [int(c) for c in contexts_sel]
                bench_cfg = BenchConfig(
                    enabled=True,
                    contexts=contexts,
                    repeats=int(repeats_val),
                    warmup=bool(warmup_val),
                    warmup_new_tokens=int(warmup_tokens_val),
                    measure_new_tokens=int(measure_tokens_val),
                    output_dir=cfg.bench.output_dir,
                )
                rows, out_dir, summary = run_benchmark(
                    registry,
                    profiles,
                    cfg.app,
                    cfg.generation_defaults,
                    bench_cfg,
                    models_sel,
                    profiles_sel,
                    contexts,
                )
                table = [[row.get(field) for field in RESULT_FIELDS] for row in rows]
                status = f"Saved to `{out_dir}`. Errors: {summary.get('num_errors', 0)}"
                return table, out_dir, status

            bench_run.click(
                _run_benchmark_ui,
                inputs=[
                    bench_models,
                    bench_profiles,
                    bench_contexts,
                    bench_repeats,
                    bench_warmup,
                    bench_warmup_tokens,
                    bench_measure_tokens,
                ],
                outputs=[bench_results, bench_output, bench_status],
            )

    return demo


def main() -> None:
    args = parse_args()
    cfg = load_root_config(args.config) if args.config else load_root_config("configs/airstudio.yaml")
    cfg = apply_overrides(cfg, args)
    ensure_offline(cfg.app)

    app = build_app(cfg, args)
    app.queue(default_concurrency_limit=cfg.app.concurrency_limit)
    app.launch(server_name=cfg.app.host, server_port=cfg.app.port, share=args.share)


if __name__ == "__main__":
    main()







