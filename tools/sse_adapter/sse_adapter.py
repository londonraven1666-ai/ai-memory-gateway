"""
Gateway SSE adapter.

Parses OpenAI-compatible chat completion SSE streams and normalizes content,
reasoning, tool_call deltas, usage, and done events. The parser accepts arbitrary
byte chunks, so it is safe to feed it raw network chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ToolCallState:
    index: int
    id: str = ""
    type: str = "function"
    function: Dict[str, str] = field(default_factory=lambda: {"name": "", "arguments": ""})


@dataclass
class SSEEvent:
    type: str
    data: Any = None
    raw: str = ""


class GatewaySSEAdapter:
    """Incremental parser for Gateway/OpenAI-compatible SSE response streams."""

    def __init__(self) -> None:
        self._buffer = ""
        self.content_parts: List[str] = []
        self.reasoning_parts: List[str] = []
        self.tool_calls: Dict[int, ToolCallState] = {}
        self.usage: Dict[str, Any] = {}
        self.done = False

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)

    def tool_call_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "index": state.index,
                "id": state.id,
                "type": state.type,
                "function": dict(state.function),
            }
            for _, state in sorted(self.tool_calls.items())
        ]

    def feed(self, chunk: bytes | str) -> List[SSEEvent]:
        text = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else chunk
        self._buffer += text
        events: List[SSEEvent] = []

        while "\n\n" in self._buffer:
            block, self._buffer = self._buffer.split("\n\n", 1)
            events.extend(self._parse_block(block))

        return events

    def close(self) -> List[SSEEvent]:
        if not self._buffer.strip():
            self._buffer = ""
            return []
        block = self._buffer
        self._buffer = ""
        return self._parse_block(block)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_call_list(),
            "usage": dict(self.usage),
            "done": self.done,
        }

    def _parse_block(self, block: str) -> List[SSEEvent]:
        data_lines = []
        for line in block.replace("\r\n", "\n").split("\n"):
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            return []

        raw_data = "\n".join(data_lines)
        if raw_data == "[DONE]":
            self.done = True
            return [SSEEvent(type="done", raw=raw_data)]

        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            return [SSEEvent(type="invalid", data=raw_data, raw=raw_data)]

        events = [SSEEvent(type="raw", data=payload, raw=raw_data)]
        if "usage" in payload and payload["usage"] is not None:
            self.usage = payload["usage"]
            events.append(SSEEvent(type="usage", data=self.usage, raw=raw_data))

        for choice in payload.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                self.content_parts.append(content)
                events.append(SSEEvent(type="content", data=content, raw=raw_data))

            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                self.reasoning_parts.append(reasoning)
                events.append(SSEEvent(type="reasoning", data=reasoning, raw=raw_data))

            for tool_call in delta.get("tool_calls") or []:
                state = self._merge_tool_call(tool_call)
                events.append(
                    SSEEvent(
                        type="tool_call",
                        data={
                            "index": state.index,
                            "id": state.id,
                            "type": state.type,
                            "function": dict(state.function),
                        },
                        raw=raw_data,
                    )
                )

            if choice.get("finish_reason"):
                events.append(SSEEvent(type="finish", data=choice["finish_reason"], raw=raw_data))

        return events

    def _merge_tool_call(self, tool_call: Dict[str, Any]) -> ToolCallState:
        index = int(tool_call.get("index", 0))
        state = self.tool_calls.get(index)
        if state is None:
            state = ToolCallState(index=index)
            self.tool_calls[index] = state

        if tool_call.get("id"):
            state.id = tool_call["id"]
        if tool_call.get("type"):
            state.type = tool_call["type"]

        fn = tool_call.get("function") or {}
        if fn.get("name"):
            state.function["name"] = fn["name"]
        if "arguments" in fn and fn["arguments"] is not None:
            state.function["arguments"] += fn["arguments"]

        return state


def parse_sse(raw: bytes | str | Iterable[bytes | str]) -> Dict[str, Any]:
    adapter = GatewaySSEAdapter()
    chunks = [raw] if isinstance(raw, (bytes, str)) else raw
    for chunk in chunks:
        adapter.feed(chunk)
    adapter.close()
    return adapter.snapshot()
