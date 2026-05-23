"""Normalize heterogeneous LLM responses into a common shape.

`expect()` accepts anything: a raw string, a dict, an OpenAI
ChatCompletion object, or an Anthropic Message. This module pulls out the
two things assertions care about — the assistant text and any tool calls —
without depending on the openai or anthropic SDKs being installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Normalized:
    """The common shape every assertion operates on."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


def _as_dict(obj: Any) -> Any:
    """Best-effort conversion of an SDK object to a plain dict.

    Handles pydantic v2 (.model_dump), pydantic v1 (.dict), and objects
    that are already dict-like. Falls back to the object unchanged.
    """
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    return obj


def _parse_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {"_value": parsed}
        except (json.JSONDecodeError, ValueError):
            return {"_raw": raw_args}
    return {}


def _extract_openai(data: dict) -> Normalized | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    tool_calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tool_calls.append(
            ToolCall(name=fn.get("name", ""), arguments=_parse_args(fn.get("arguments")))
        )
    return Normalized(text=text or "", tool_calls=tool_calls, raw=data)


def _extract_anthropic(data: dict) -> Normalized | None:
    content = data.get("content")
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(name=block.get("name", ""), arguments=block.get("input") or {})
            )
    return Normalized(text="".join(text_parts), tool_calls=tool_calls, raw=data)


def normalize(response: Any) -> Normalized:
    """Turn any supported response into a Normalized object."""
    if isinstance(response, Normalized):
        return response
    if isinstance(response, str):
        return Normalized(text=response, raw=response)

    data = _as_dict(response)
    if isinstance(data, dict):
        # Try OpenAI shape first (has "choices"), then Anthropic ("content").
        for extractor in (_extract_openai, _extract_anthropic):
            result = extractor(data)
            if result is not None:
                return result
        # A bare dict with a "text" or "content" string field.
        for key in ("text", "content", "output_text"):
            val = data.get(key)
            if isinstance(val, str):
                return Normalized(text=val, raw=data)

    # Last resort: stringify. Better than throwing — lets substring/regex
    # matchers still work on something.
    return Normalized(text=str(response), raw=response)
