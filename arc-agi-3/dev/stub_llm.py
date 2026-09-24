#!/usr/bin/env python3
"""Stub OpenAI-compatible server that plays a fixed action script.

Each /v1/chat/completions call returns a python tool_call. The script
repeats the same MOUSE click so the no-op/volatile-cell code paths fire.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

CODE = (
    "action([{'action': 'MOUSE', 'row': 3, 'col': 3}, "
    "{'action': 'MOUSE', 'row': 3, 'col': 3}, "
    "{'action': 'MOUSE', 'row': 3, 'col': 3}])"
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            req = json.loads(body)
        except Exception:
            req = {}
        assert self.path.endswith("/chat/completions"), self.path
        resp = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_stub",
                                "type": "function",
                                "function": {
                                    "name": "python",
                                    "arguments": json.dumps({"code": CODE}),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


HTTPServer(("127.0.0.1", 18080), Handler).serve_forever()
