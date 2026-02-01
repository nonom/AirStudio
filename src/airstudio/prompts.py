"""Prompt builders."""
from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM = "You are a helpful assistant."


def build_chat_messages(history: list[tuple[str, str]], user_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})
    return messages


def build_messages_for_context(tokenizer: Any, family_hint: str | None, target_tokens: int) -> list[dict[str, str]]:
    base = "Answer with a short summary."
    pad = " lorem ipsum"
    text = base

    def count_tokens(msgs: list[dict[str, str]]) -> int:
        prompt = _render_prompt(tokenizer, msgs)
        return len(tokenizer(prompt)["input_ids"])

    messages = [{"role": "system", "content": DEFAULT_SYSTEM}, {"role": "user", "content": text}]
    current = count_tokens(messages)
    while current < target_tokens - 16:
        text += pad
        messages[1]["content"] = text
        current = count_tokens(messages)
    return messages


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    lines = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        lines.append(f"{role}: {msg.get('content','')}")
    lines.append("Assistant:")
    return "\n".join(lines)
