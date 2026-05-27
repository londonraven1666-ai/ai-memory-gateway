import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sse_adapter import GatewaySSEAdapter, parse_sse


class GatewaySSEAdapterTests(unittest.TestCase):
    def test_plain_content_stream(self):
        raw = (
            'data: {"choices":[{"delta":{"role":"assistant"},"index":0}]}\n\n'
            'data: {"choices":[{"delta":{"content":"hello"},"index":0}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"},"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
            "data: [DONE]\n\n"
        )

        parsed = parse_sse(raw)

        self.assertEqual(parsed["content"], "hello world")
        self.assertEqual(parsed["reasoning"], "")
        self.assertEqual(parsed["tool_calls"], [])
        self.assertEqual(parsed["usage"]["total_tokens"], 5)
        self.assertTrue(parsed["done"])

    def test_reasoning_content_stream(self):
        raw = (
            'data: {"choices":[{"delta":{"reasoning_content":"think "},"index":0}]}\n\n'
            'data: {"choices":[{"delta":{"reasoning_content":"briefly","content":"4"},"index":0}]}\n\n'
            "data: [DONE]\n\n"
        )

        parsed = parse_sse(raw)

        self.assertEqual(parsed["reasoning"], "think briefly")
        self.assertEqual(parsed["content"], "4")
        self.assertTrue(parsed["done"])

    def test_fragmented_tool_call_stream(self):
        raw = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"capture_probe","arguments":"{\\"status\\""}}]},"index":0}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"ok\\",\\"count\\":1}"}}]},"index":0,"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )

        parsed = parse_sse(raw)

        self.assertEqual(len(parsed["tool_calls"]), 1)
        call = parsed["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["function"]["name"], "capture_probe")
        self.assertEqual(call["function"]["arguments"], '{"status":"ok","count":1}')
        self.assertTrue(parsed["done"])

    def test_arbitrary_network_chunk_boundaries(self):
        adapter = GatewaySSEAdapter()
        events = []
        for chunk in [
            b'data: {"choices":[{"delta":{"content":"hel',
            b'lo"},"index":0}]}\n',
            b"\n",
            b"data: [DONE]\n\n",
        ]:
            events.extend(adapter.feed(chunk))
        events.extend(adapter.close())

        self.assertEqual(adapter.content, "hello")
        self.assertEqual(events[-1].type, "done")


if __name__ == "__main__":
    unittest.main()
