# Gateway SSE Adapter Reference

This directory contains the Python reference implementation for the future Tinte
TypeScript SSE adapter. It is also the Gateway-side capture tool for recording
real raw SSE events from the public Gateway endpoint.

## Files

- `sse_adapter.py` parses OpenAI-compatible SSE chunks into normalized content,
  reasoning, tool call deltas, usage, and done state.
- `capture_gateway_sse.py` captures raw Gateway SSE fixtures for three scenarios:
  plain content, reasoning, and tool calls.
- `tests/test_sse_adapter.py` covers the parser behavior with local fixtures.

## Run Tests

From the repository root:

```bash
python3 -m unittest tools/sse_adapter/tests/test_sse_adapter.py
```

## Capture Real Gateway SSE

Do not SSH into the VPS to obtain credentials. Set `GATEWAY_API_KEY` locally
after Aquila provides it, then run:

```bash
GATEWAY_API_KEY="..." python3 tools/sse_adapter/capture_gateway_sse.py
```

Optional environment variables:

- `GATEWAY_URL`, default:
  `https://altairaquila-hafen.duckdns.org/gateway/v1/chat/completions`
- `GATEWAY_MODEL`, default: `gateway`

The capture script writes `.sse` raw event files and small `.json` metadata
files to `tools/sse_adapter/fixtures/gateway_sse/`.

After capture, document these details in the PR description:

- the actual thinking/reasoning field name
- the streamed `tool_calls` delta structure
- where `usage` appears in the stream

## Captured Gateway Observations

Captured from the public Gateway endpoint on 2026-05-27:

- Thinking/reasoning arrives as `choices[].delta.reasoning_content`.
- Tool calls arrive as streamed `choices[].delta.tool_calls[]` fragments keyed by
  `index`. The first fragment includes `id`, `type`, `function.name`, and an
  empty or partial `function.arguments`; later fragments usually include the
  same `index` plus more `function.arguments`. Consumers must concatenate
  `function.arguments` by `index`.
- Usage appears in a separate event after the finish event and before
  `data: [DONE]`. That event has `choices: []` and a top-level `usage` object.
  Intermediate events carry `"usage": null`.
