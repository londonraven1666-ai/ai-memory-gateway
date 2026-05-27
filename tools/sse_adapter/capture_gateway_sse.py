#!/usr/bin/env python3
"""
Capture raw Gateway SSE fixtures for three scenarios:
  1. plain content
  2. reasoning-capable request
  3. streamed tool call

Requires:
  GATEWAY_API_KEY

Optional:
  GATEWAY_URL   default https://altairaquila-hafen.duckdns.org/gateway/v1/chat/completions
  GATEWAY_MODEL default gateway
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
from urllib import request, error


DEFAULT_URL = "https://altairaquila-hafen.duckdns.org/gateway/v1/chat/completions"


def request_body(name: str, model: str) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "model": model,
        "stream": True,
        "max_tokens": 160,
    }
    if name == "plain":
        return {
            **base,
            "messages": [{"role": "user", "content": "用一句话回答：SSE plain capture OK"}],
        }
    if name == "reasoning":
        return {
            **base,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "先简短思考，再用一句话回答：2+2 等于几？"}],
        }
    if name == "tool_call":
        return {
            **base,
            "tool_choice": {"type": "function", "function": {"name": "capture_probe"}},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "capture_probe",
                        "description": "Return a tiny JSON object for SSE tool call capture.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string"},
                                "count": {"type": "integer"},
                            },
                            "required": ["status", "count"],
                        },
                    },
                }
            ],
            "messages": [{"role": "user", "content": "Call capture_probe with status ok and count 1."}],
        }
    raise ValueError(f"unknown scenario: {name}")


def capture(url: str, api_key: str, model: str, scenario: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{scenario}.sse"
    body = request_body(scenario, model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        response = request.urlopen(req, timeout=120)
    except error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} while capturing {scenario}: {body_text[:500]}")

    with response:
        with output.open("wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)

    meta = {
        "scenario": scenario,
        "url": url,
        "model": model,
        "request": {k: v for k, v in body.items() if k != "messages"},
        "output": output.name,
    }
    (out_dir / f"{scenario}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"captured {scenario}: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["plain", "reasoning", "tool_call", "all"], default="all")
    default_out_dir = Path(__file__).resolve().parent / "fixtures" / "gateway_sse"
    parser.add_argument("--out-dir", default=str(default_out_dir))
    args = parser.parse_args()

    api_key = os.getenv("GATEWAY_API_KEY")
    if not api_key:
        raise SystemExit("GATEWAY_API_KEY is required")

    url = os.getenv("GATEWAY_URL", DEFAULT_URL)
    model = os.getenv("GATEWAY_MODEL", "gateway")
    scenarios = ["plain", "reasoning", "tool_call"] if args.scenario == "all" else [args.scenario]
    out_dir = Path(args.out_dir)
    for scenario in scenarios:
        capture(url, api_key, model, scenario, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
