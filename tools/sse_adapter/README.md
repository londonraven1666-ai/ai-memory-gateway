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
