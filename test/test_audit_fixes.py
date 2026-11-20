"""Negative checks for C1, C4, and C5.

Bad client requests must stop before the mock model.
Upstream error bodies must come back as short JSON, without the prompt.
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


SECRET = "SECRET-PROMPT-XYZ"
CHAT = b'{"messages":[{"role":"user","content":"hi"}]}'


class CountingUpstream(BaseHTTPRequestHandler):
    hits = 0
    status = 200
    payload = b'{"message":{"role":"assistant","content":"ok"}}'
    last = b""

    def _reply(self) -> None:
        CountingUpstream.hits += 1
        body = self.payload
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        CountingUpstream.last = self.rfile.read(length) if length else b""
        self._reply()

    def do_GET(self) -> None:
        self._reply()

    def log_message(self, fmt: str, *args) -> None:
        return


def listen(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def stop(httpd, thread) -> None:
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def raw_call(port: int, method: str, path: str, body: bytes | None = None, headers=None, timeout: float = 3):
    fields = dict(headers or {})
    host = fields.pop("Host", f"127.0.0.1:{port}")
    extra_lines = fields.pop("Extra", [])
    lines = [f"{method} {path} HTTP/1.1", f"Host: {host}"]
    payload = body or b""
    if body is not None and "Content-Length" not in fields:
        lines.append(f"Content-Length: {len(payload)}")
    for name, value in fields.items():
        lines.append(f"{name}: {value}")
    for extra in extra_lines:
        lines.append(extra)
    lines.append("Connection: close")
    blob = ("\r\n".join(lines) + "\r\n\r\n").encode("iso-8859-1")
    if body is not None:
        blob += payload
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.settimeout(timeout)
    try:
        sock.sendall(blob)
        chunks = []
        while True:
            try:
                part = sock.recv(8192)
            except socket.timeout:
                break
            if not part:
                break
            chunks.append(part)
    finally:
        sock.close()
    raw = b"".join(chunks)
    if not raw:
        return None, {}, b""
    head, sep, resp = raw.partition(b"\r\n\r\n")
    if not sep:
        return None, {}, raw
    first, *rest = head.decode("iso-8859-1", "replace").split("\r\n")
    try:
        status = int(first.split()[1])
    except (IndexError, ValueError):
        status = None
    parsed = {}
    for line in rest:
        if ":" in line:
            key, value = line.split(":", 1)
            parsed[key.strip().lower()] = value.strip()
    length = parsed.get("content-length")
    if length and length.isdigit():
        resp = resp[: int(length)]
    return status, parsed, resp


class GuardFixture(unittest.TestCase):
    def setUp(self) -> None:
        CountingUpstream.hits = 0
        CountingUpstream.last = b""
        CountingUpstream.status = 200
        CountingUpstream.payload = b'{"message":{"role":"assistant","content":"ok"}}'
        self.upstream, self.upstream_thread = listen(CountingUpstream)
        self.config = server.Config(
            ollama_base=f"http://127.0.0.1:{self.upstream.server_address[1]}",
            model="qwen2.5:1.5b",
            host="127.0.0.1",
            port=8765,
            root=ROOT,
        )
        self.proxy, self.proxy_thread = listen(server.make_handler(self.config))
        self.port = self.proxy.server_address[1]

    def tearDown(self) -> None:
        stop(self.proxy, self.proxy_thread)
        stop(self.upstream, self.upstream_thread)

    def call(self, method, path, body=None, headers=None):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status, response_headers, response_body = raw_call(
                self.port, method, path, body, headers
            )
        self.stderr = stderr.getvalue()
        return status, response_headers, response_body


class LocalRequestTests(GuardFixture):
    def test_foreign_host_does_not_reach_upstream(self) -> None:
        for host in (
            "evil.example",
            f"evil.example:{self.port}",
            f"127.0.0.1.evil.example:{self.port}",
            "127.0.0.1:9",
            f"localhost:{self.port + 1}",
        ):
            with self.subTest(host=host):
                CountingUpstream.hits = 0
                status, _, body = self.call(
                    "POST",
                    "/api/chat",
                    CHAT,
                    {"Host": host, "Content-Type": "application/json"},
                )
                self.assertEqual(status, 403)
                self.assertEqual(json.loads(body)["error"], "forbidden")
                self.assertNotIn(b"evil", body)
                self.assertEqual(CountingUpstream.hits, 0)

    def test_foreign_origin_does_not_reach_upstream(self) -> None:
        for origin in (
            "https://evil.example",
            "null",
            f"http://127.0.0.1:{self.port}/",
            f"http://user@127.0.0.1:{self.port}",
            f"https://127.0.0.1:{self.port}",
            "http://127.0.0.1:9",
        ):
            with self.subTest(origin=origin):
                CountingUpstream.hits = 0
                status, _, body = self.call(
                    "POST",
                    "/api/chat",
                    CHAT,
                    {
                        "Host": f"127.0.0.1:{self.port}",
                        "Origin": origin,
                        "Content-Type": "application/json",
                        "Sec-Fetch-Site": "cross-site",
                    },
                )
                self.assertEqual(status, 403)
                self.assertEqual(json.loads(body)["error"], "forbidden")
                self.assertEqual(CountingUpstream.hits, 0)
                self.assertNotIn(SECRET.encode(), body)

    def test_two_origin_headers_are_forbidden(self) -> None:
        good = f"http://127.0.0.1:{self.port}"
        bad = "https://evil.example"
        for first, second in ((bad, good), (good, bad)):
            with self.subTest(first=first, second=second):
                CountingUpstream.hits = 0
                status, _, body = self.call(
                    "POST",
                    "/api/chat",
                    CHAT,
                    {
                        "Host": f"127.0.0.1:{self.port}",
                        "Origin": first,
                        "Content-Type": "application/json",
                        "Extra": [f"Origin: {second}"],
                    },
                )
                self.assertEqual(status, 403)
                self.assertEqual(json.loads(body)["error"], "forbidden")
                self.assertEqual(CountingUpstream.hits, 0)

    def test_dotted_localhost_host_is_forbidden(self) -> None:
        status, _, body = self.call(
            "POST",
            "/api/chat",
            CHAT,
            {"Host": "localhost.", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["error"], "forbidden")
        self.assertEqual(CountingUpstream.hits, 0)

    def test_ipv6_loopback_host_is_forbidden(self) -> None:
        status, _, body = self.call(
            "POST",
            "/api/chat",
            CHAT,
            {
                "Host": f"[::1]:{self.port}",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["error"], "forbidden")
        self.assertEqual(CountingUpstream.hits, 0)

    def test_simple_content_types_do_not_reach_upstream(self) -> None:
        for content_type in (
            None,
            "text/plain",
            "application/x-www-form-urlencoded",
            "text/json",
        ):
            with self.subTest(content_type=content_type):
                CountingUpstream.hits = 0
                headers = {"Host": f"127.0.0.1:{self.port}"}
                if content_type is not None:
                    headers["Content-Type"] = content_type
                status, _, body = self.call("POST", "/api/chat", CHAT, headers)
                self.assertEqual(status, 415)
                self.assertEqual(json.loads(body)["error"], "unsupported type")
                self.assertEqual(CountingUpstream.hits, 0)

    def test_two_host_headers_are_forbidden(self) -> None:
        status, _, body = self.call(
            "POST",
            "/api/chat",
            CHAT,
            {
                "Host": f"127.0.0.1:{self.port}",
                "Content-Type": "application/json",
                "Extra": ["Host: evil.example"],
            },
        )
        self.assertEqual(status, 403)
        self.assertEqual(CountingUpstream.hits, 0)
        self.assertNotIn(b"evil.example", body)

    def test_config_is_hidden_from_a_foreign_host(self) -> None:
        status, _, body = self.call(
            "GET",
            "/api/config",
            headers={"Host": f"rebind.example:{self.port}"},
        )
        self.assertEqual(status, 403)
        self.assertNotIn(b"ollama_base", body)
        self.assertNotIn(str(self.upstream.server_address[1]).encode(), body)
        self.assertEqual(CountingUpstream.hits, 0)

    def test_health_is_not_probed_for_a_foreign_origin(self) -> None:
        status, _, body = self.call(
            "GET",
            "/api/health",
            headers={
                "Host": f"127.0.0.1:{self.port}",
                "Origin": "https://evil.example",
            },
        )
        self.assertEqual(status, 403)
        self.assertEqual(CountingUpstream.hits, 0)
        self.assertNotIn(b"ok", body)

    def test_browser_like_chat_still_works(self) -> None:
        pairs = (
            (f"127.0.0.1:{self.port}", f"http://127.0.0.1:{self.port}"),
            (f"localhost:{self.port}", f"http://localhost:{self.port}"),
        )
        for host, origin in pairs:
            with self.subTest(host=host):
                CountingUpstream.hits = 0
                status, headers, body = self.call(
                    "POST",
                    "/api/chat",
                    b'{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"hi"}]}',
                    {
                        "Host": host,
                        "Origin": origin,
                        "Content-Type": "application/json; charset=utf-8",
                        "X-Request-Id": "4",
                        "Sec-Fetch-Site": "same-origin",
                        "Sec-Fetch-Mode": "cors",
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(headers.get("x-request-id"), "4")
                self.assertEqual(json.loads(body)["message"]["content"], "ok")
                self.assertEqual(CountingUpstream.hits, 1)

    def test_portless_host_still_works(self) -> None:
        status, _, body = self.call(
            "POST",
            "/api/chat",
            CHAT,
            {"Host": "127.0.0.1", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["message"]["content"], "ok")
        self.assertEqual(CountingUpstream.hits, 1)

    def test_missing_origin_still_works(self) -> None:
        status, headers, body = self.call(
            "POST",
            "/api/chat",
            CHAT,
            {
                "Host": f"127.0.0.1:{self.port}",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 200)
        self.assertIsNone(headers.get("access-control-allow-origin"))
        self.assertEqual(json.loads(body)["message"]["content"], "ok")
        self.assertEqual(CountingUpstream.hits, 1)

    def test_config_port_is_not_the_listen_port(self) -> None:
        if self.port == 8765:
            self.skipTest("ephemeral port collided with 8765")
        status, _, _ = self.call(
            "POST",
            "/api/chat",
            CHAT,
            {"Host": "127.0.0.1:8765", "Content-Type": "application/json"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(CountingUpstream.hits, 0)

    def test_cross_site_navigation_can_open_the_page(self) -> None:
        status, headers, body = self.call(
            "GET",
            "/",
            headers={
                "Host": f"localhost:{self.port}",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Dest": "document",
            },
        )
        self.assertEqual(status, 200)
        self.assertIn(b"text/html", headers.get("content-type", "").encode())
        self.assertIn("Отправить".encode(), body)


class BadClientJsonTests(GuardFixture):
    def _post(self, body: bytes):
        return self.call(
            "POST",
            "/api/chat",
            body,
            {"Host": f"127.0.0.1:{self.port}", "Content-Type": "application/json"},
        )

    def test_wrong_message_container_is_a_bad_message(self) -> None:
        for raw in (
            b'{"messages":"hi"}',
            b'{"messages":{"role":"user","content":"hi"}}',
            b'{"messages":1}',
            b'{"messages":true}',
        ):
            with self.subTest(raw=raw):
                CountingUpstream.hits = 0
                status, _, body = self._post(raw)
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body)["error"], "bad message")
                self.assertEqual(CountingUpstream.hits, 0)

    def test_null_model_uses_the_server_default(self) -> None:
        status, _, body = self._post(
            b'{"model":null,"messages":[{"role":"user","content":"hi"}]}'
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["message"]["content"], "ok")
        self.assertEqual(CountingUpstream.hits, 1)
        sent = json.loads(CountingUpstream.last)
        self.assertEqual(sent["model"], "qwen2.5:1.5b")

    def test_null_and_empty_messages_stay_required(self) -> None:
        for raw in (b"{}", b'{"messages":null}', b'{"messages":[]}'):
            with self.subTest(raw=raw):
                status, _, body = self._post(raw)
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body)["error"], "messages required")
                self.assertEqual(CountingUpstream.hits, 0)

    def test_element_type_errors_stay_clear(self) -> None:
        samples = {
            b'{"messages":[{"role":"user","content":1}]}': "bad message",
            b'{"messages":[{"role":1,"content":"hi"}]}': "bad message",
            b'{"model":1,"messages":[{"role":"user","content":"hi"}]}': "bad model",
            b"{": "bad json",
            b"[]": "bad json",
        }
        for raw, code in samples.items():
            with self.subTest(raw=raw):
                CountingUpstream.hits = 0
                status, _, body = self._post(raw)
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body)["error"], code)
                self.assertEqual(CountingUpstream.hits, 0)

    def test_lone_surrogate_is_a_bad_message(self) -> None:
        raw = '{"messages":[{"role":"user","content":"\\ud800%s"}]}' % SECRET
        status, _, body = self._post(raw.encode())
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "bad message")
        self.assertNotIn(b"utf-8", body)
        self.assertNotIn(SECRET.encode(), body)
        self.assertEqual(CountingUpstream.hits, 0)

    def test_deep_json_is_bad_json_not_a_crash(self) -> None:
        raw = b"[" * 30000 + b"0" + b"]" * 30000
        try:
            server.parse_object(raw)
        except RecursionError:
            self.fail("deep json escaped as RecursionError")
        except ValueError as err:
            self.assertEqual(str(err), "bad json")
        else:
            self.fail("deep json was accepted")
        status, _, body = self._post(raw)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "bad json")
        self.assertNotIn("Traceback", self.stderr)
        self.assertEqual(CountingUpstream.hits, 0)


class UpstreamErrorTests(GuardFixture):
    def _chat(self):
        return self.call(
            "POST",
            "/api/chat",
            CHAT,
            {"Host": f"127.0.0.1:{self.port}", "Content-Type": "application/json"},
        )

    def test_plain_upstream_error_hides_the_prompt(self) -> None:
        CountingUpstream.status = 500
        CountingUpstream.payload = f"plain text blew up {SECRET}".encode()
        status, headers, body = self._chat()
        self.assertEqual(status, 500)
        self.assertIn("application/json", headers.get("content-type", ""))
        self.assertEqual(json.loads(body)["error"], "unreachable")
        self.assertNotIn(SECRET.encode(), body)
        self.assertEqual(CountingUpstream.hits, 1)

    def test_object_upstream_error_hides_the_prompt(self) -> None:
        CountingUpstream.status = 500
        CountingUpstream.payload = json.dumps(
            {"error": {"message": "nope", "prompt": SECRET}}
        ).encode()
        status, _, body = self._chat()
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body), {"error": "unreachable"})
        self.assertNotIn(SECRET.encode(), body)

    def test_model_not_found_is_kept_without_extra_fields(self) -> None:
        CountingUpstream.status = 404
        CountingUpstream.payload = json.dumps(
            {"error": "model not found", "prompt": SECRET}
        ).encode()
        status, _, body = self._chat()
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body), {"error": "model not found"})
        self.assertNotIn(SECRET.encode(), body)

    def test_model_id_error_is_kept_and_a_sentence_is_not(self) -> None:
        CountingUpstream.status = 404
        CountingUpstream.payload = b'{"error":"model \'qwen2.5:1.5b\' not found"}'
        status, _, body = self._chat()
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"], "model 'qwen2.5:1.5b' not found")
        CountingUpstream.hits = 0
        CountingUpstream.payload = json.dumps(
            {"error": f"please repeat {SECRET}"}
        ).encode()
        status, _, body = self._chat()
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"], "unreachable")
        self.assertNotIn(SECRET.encode(), body)

    def test_non_json_success_is_not_forwarded(self) -> None:
        CountingUpstream.status = 200
        CountingUpstream.payload = f"<html>{SECRET}</html>".encode()
        status, headers, body = self._chat()
        self.assertEqual(status, 502)
        self.assertIn("application/json", headers.get("content-type", ""))
        self.assertEqual(json.loads(body)["error"], "unreachable")
        self.assertNotIn(SECRET.encode(), body)

    def test_non_string_content_is_not_forwarded(self) -> None:
        CountingUpstream.status = 200
        CountingUpstream.payload = json.dumps(
            {"message": {"role": "assistant", "content": 5}, "prompt": SECRET}
        ).encode()
        status, _, body = self._chat()
        self.assertEqual(status, 502)
        self.assertEqual(json.loads(body)["error"], "unreachable")
        self.assertNotIn(SECRET.encode(), body)

    def test_health_object_error_stays_json(self) -> None:
        CountingUpstream.status = 401
        CountingUpstream.payload = json.dumps({"error": {"message": SECRET}}).encode()
        status, _, body = self.call(
            "GET",
            "/api/health",
            headers={"Host": f"127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"proxy": "ok", "backend": "unreachable"})
        self.assertNotIn(SECRET.encode(), body)
        self.assertNotIn("Traceback", self.stderr)
        self.assertEqual(CountingUpstream.hits, 1)


class LmStudioErrorTests(unittest.TestCase):
    def test_object_error_does_not_crash(self) -> None:
        status, body = server.lmstudio_reply(
            400, json.dumps({"error": {"message": SECRET}}).encode()
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "unreachable"})
        self.assertNotIn(SECRET.encode(), body)

    def test_http_object_error_hides_the_prompt(self) -> None:
        CountingUpstream.hits = 0
        CountingUpstream.status = 400
        CountingUpstream.payload = json.dumps({"error": {"message": SECRET}}).encode()
        upstream, upstream_thread = listen(CountingUpstream)
        proxy, proxy_thread = listen(
            server.make_handler(
                server.Config(
                    ollama_base="",
                    model="qwen/qwen3.5-9b",
                    host="127.0.0.1",
                    port=8765,
                    root=ROOT,
                    provider="lmstudio",
                    lmstudio_base=f"http://127.0.0.1:{upstream.server_address[1]}",
                )
            )
        )
        port = proxy.server_address[1]
        stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(stderr):
                status, _, body = raw_call(
                    port,
                    "POST",
                    "/api/chat",
                    CHAT,
                    {
                        "Host": f"127.0.0.1:{port}",
                        "Origin": f"http://127.0.0.1:{port}",
                        "Content-Type": "application/json",
                    },
                )
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body), {"error": "unreachable"})
        self.assertNotIn(SECRET.encode(), body)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(CountingUpstream.hits, 1)


if __name__ == "__main__":
    unittest.main()
