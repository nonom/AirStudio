"""AirLLM engine implementation."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from airllm import AutoModel

from .base import DeviceSpec, GenerationOutput, GenerationSpec
from ..prompts import _render_prompt


def _ensure_safetensors_index(model_path: str) -> None:
    index_path = Path(model_path) / "model.safetensors.index.json"
    if index_path.exists():
        return
    st_path = Path(model_path) / "model.safetensors"
    if not st_path.exists():
        return
    try:
        from safetensors import safe_open
    except Exception:
        return
    weight_map: dict[str, str] = {}
    with safe_open(str(st_path), framework="pt") as f:
        for key in f.keys():
            weight_map[key] = st_path.name
    data = {
        "metadata": {"total_size": os.path.getsize(st_path)},
        "weight_map": weight_map,
    }
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def _is_mistral_model(model: Any) -> bool:
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "model_type", None) == "mistral":
        return True
    return "mistral" in model.__class__.__name__.lower()


def _patch_mistral_position_embeddings(model: Any) -> None:
    if not _is_mistral_model(model):
        return
    if not hasattr(model, "get_pos_emb_args"):
        return
    base_model = getattr(model, "model", None)
    rotary = None
    if base_model is not None:
        rotary = getattr(base_model, "rotary_emb", None)
        if rotary is None:
            inner = getattr(base_model, "model", None)
            if inner is not None:
                rotary = getattr(inner, "rotary_emb", None)
    if rotary is None:
        return
    original = model.get_pos_emb_args

    def _get_pos_emb_args(len_p: int, len_s: int):
        if len_s <= 0:
            return original(len_p, len_s)
        try:
            device = getattr(model, "running_device", None) or getattr(model, "device", None) or "cpu"
            dtype = getattr(model, "running_dtype", torch.float16)
            position_ids = torch.arange(len_p, len_p + len_s, device=device, dtype=torch.long).unsqueeze(0)
            dummy = torch.empty((1,), device=device, dtype=dtype)
            cos, sin = rotary(dummy, position_ids)
            return {"position_embeddings": (cos, sin)}
        except Exception:
            return original(len_p, len_s)

    model.get_pos_emb_args = _get_pos_emb_args


class AirLLMEngine:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: torch.device | None = None

    def load(self, model_path: str, compression: str | None, layer_cache_dir: str, device: DeviceSpec) -> None:
        if device.kind == "cuda" and torch.cuda.is_available():
            index = device.gpu_index if device.gpu_index is not None else 0
            self._device = torch.device(f"cuda:{index}")
        else:
            self._device = torch.device("cpu")

        if os.path.isdir(model_path):
            _ensure_safetensors_index(model_path)

        if layer_cache_dir:
            os.makedirs(layer_cache_dir, exist_ok=True)

        self._model = AutoModel.from_pretrained(
            model_path,
            layer_shards_saving_path=layer_cache_dir,
            compression=compression,
        )
        _patch_mistral_position_embeddings(self._model)

        self._tokenizer = getattr(self._model, "tokenizer", None)
        if self._tokenizer is None:
            raise RuntimeError("Model tokenizer not available")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tokenizer(self):
        return self._tokenizer

    def generate(self, messages: list[dict], gen: GenerationSpec, family_hint: str | None) -> GenerationOutput:
        if self._model is None or self._tokenizer is None or self._device is None:
            raise RuntimeError("Engine not loaded")

        prompt = _render_prompt(self._tokenizer, messages)
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=gen.max_context,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self._device)

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        start = time.perf_counter()
        output_ids = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p,
            do_sample=gen.do_sample,
            use_cache=False,
        )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        end = time.perf_counter()

        if isinstance(output_ids, (list, tuple)):
            if len(output_ids) == 0:
                raise RuntimeError("Empty output from model")
            output_ids = output_ids[0]
        if output_ids.ndim == 1:
            output_ids = output_ids.unsqueeze(0)

        prompt_tokens = int(input_ids.shape[-1])
        total_tokens = int(output_ids.shape[-1])
        generated_tokens = max(0, total_tokens - prompt_tokens)
        text = self._tokenizer.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)

        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            decode_time_s=end - start,
        )



