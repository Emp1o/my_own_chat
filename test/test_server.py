"""HTTP tests for the local chat proxy.

The checks speak HTTP. They do not call the handler's methods.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import server  # noqa: E402


def http_call(port: int, path: str, body: bytes | None = None, timeout: float = 5):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        method="POST" if body is not None else "GET",
    )
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type", ""), err.read()


def listen(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def stop(httpd, thread) -> None:
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def assistant_payload(text: str) -> bytes:
    return json.dumps({"message": {"role": "assistant", "content": text}}).encode()


class RecordingOllama(BaseHTTPRequestHandler):
    status = 200
    payload = assistant_payload("привет")
    last = b""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        RecordingOllama.last = self.rfile.read(length) if length else b""
        body = self.payload
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


class ProxyFixture:
    def setUp(self) -> None:
        RecordingOllama.status = 200
        RecordingOllama.payload = assistant_payload("привет")
        RecordingOllama.last = b""
        self.upstream, self.upstream_thread = listen(RecordingOllama)
        port = self.upstream.server_address[1]
        self.config = server.Config(
            ollama_base=f"http://127.0.0.1:{port}",
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


class ProxyTests(ProxyFixture, unittest.TestCase):
    def test_home_page_is_the_chat(self) -> None:
        status, content_type, body = http_call(self.port, "/")
        text = body.decode("utf-8")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn("Отправить", text)
        self.assertIn("<textarea", text)
        self.assertIn('href="app.css"', text)

    def test_styles_are_served(self) -> None:
        for path in ("/app.css", "/tokens.css", "/base.css"):
            status, content_type, body = http_call(self.port, path)
            self.assertEqual(status, 200, path)
            self.assertIn("text/css", content_type, path)
            self.assertTrue(body, path)

    def test_config_reports_the_home_model(self) -> None:
        status, content_type, body = http_call(self.port, "/api/config?from=page")
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        data = json.loads(body)
        self.assertEqual(data["model"], "qwen2.5:1.5b")
        self.assertEqual(data["ollama_base"], self.config.ollama_base)

    def test_unknown_path_is_not_the_source(self) -> None:
        status, _, body = http_call(self.port, "/server.py")
        self.assertEqual(status, 404)
        self.assertNotIn(b"OLLAMA_BASE", body)

    def test_chat_is_forwarded_without_streaming(self) -> None:
        payload = json.dumps(
            {"messages": [{"role": "user", "content": "ты здесь?"}]},
            ensure_ascii=False,
        ).encode()
        status, content_type, body = http_call(self.port, "/api/chat", payload)
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(json.loads(body)["message"]["content"], "привет")
        sent = json.loads(RecordingOllama.last)
        self.assertEqual(sent["model"], "qwen2.5:1.5b")
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["messages"][0]["content"], "ты здесь?")

    def test_request_model_overrides_the_default(self) -> None:
        payload = json.dumps(
            {
                "model": "qwen2.5:7b",
                "messages": [{"role": "user", "content": "ещё"}],
            }
        ).encode()
        status, _, _ = http_call(self.port, "/api/chat", payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(RecordingOllama.last)["model"], "qwen2.5:7b")

    def test_bad_json_is_refused(self) -> None:
        status, _, body = http_call(self.port, "/api/chat", b"{")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "bad json")

    def test_bytes_that_are_not_text_are_refused(self) -> None:
        status, _, body = http_call(self.port, "/api/chat", b"\xff")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "bad json")

    def test_non_object_json_is_refused(self) -> None:
        status, _, body = http_call(self.port, "/api/chat", b"[]")
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "bad json")

    def test_missing_messages_are_refused(self) -> None:
        for raw in (b"{}", b'{"messages":[]}', b""):
            status, _, body = http_call(self.port, "/api/chat", raw)
            self.assertEqual(status, 400, raw)
            self.assertEqual(json.loads(body)["error"], "messages required")

    def test_other_post_is_missing(self) -> None:
        status, _, _ = http_call(self.port, "/api/config", b"{}")
        self.assertEqual(status, 404)

    def test_upstream_empty_error_still_answers(self) -> None:
        RecordingOllama.status = 500
        RecordingOllama.payload = b""
        payload = b'{"messages":[{"role":"user","content":"x"}]}'
        status, _, body = http_call(self.port, "/api/chat", payload)
        self.assertEqual(status, 500)
        self.assertTrue(body)

    def test_upstream_http_error_is_passed_through(self) -> None:
        RecordingOllama.status = 404
        RecordingOllama.payload = b'{"error":"model not found"}'
        payload = b'{"messages":[{"role":"user","content":"x"}]}'
        status, _, body = http_call(self.port, "/api/chat", payload)
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(body)["error"], "model not found")

    def test_other_network_failures_stay_generic(self) -> None:
        self.assertEqual(server.failure_code(OSError("network is unreachable")), "unreachable")

    def test_socket_timeout_is_its_own_status(self) -> None:
        self.assertEqual(server.failure_code(TimeoutError("timed out")), "timeout")

    def test_dead_upstream_is_a_bad_gateway(self) -> None:
        stop(self.upstream, self.upstream_thread)
        payload = b'{"messages":[{"role":"user","content":"x"}]}'
        status, content_type, body = http_call(self.port, "/api/chat", payload)
        self.assertEqual(status, 502)
        self.assertIn("application/json", content_type)
        data = json.loads(body)
        self.assertEqual(data["error"], "refused")
        self.assertNotIn("urlopen", body.decode())


class MissingFileTests(unittest.TestCase):
    def test_missing_home_file_is_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = server.Config(
                ollama_base="http://127.0.0.1:9",
                model="qwen2.5:1.5b",
                host="127.0.0.1",
                port=8765,
                root=Path(tmp),
            )
            httpd, thread = listen(server.make_handler(cfg))
            try:
                status, _, body = http_call(httpd.server_address[1], "/")
            finally:
                stop(httpd, thread)
        self.assertEqual(status, 404)
        self.assertEqual(body, b"Not found")


class EnvTests(unittest.TestCase):
    def test_env_sets_the_windows_address(self) -> None:
        saved = {
            key: os.environ.get(key)
            for key in ("OLLAMA_BASE", "OLLAMA_MODEL", "CHAT_HOST", "CHAT_PORT")
        }
        os.environ["OLLAMA_BASE"] = "http://100.64.1.8:11434/"
        os.environ["OLLAMA_MODEL"] = "qwen2.5:1.5b"
        os.environ["CHAT_HOST"] = "127.0.0.1"
        os.environ["CHAT_PORT"] = "8765"
        try:
            cfg = server.config_from_env()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.assertEqual(cfg.ollama_base, "http://100.64.1.8:11434")
        self.assertEqual(cfg.model, "qwen2.5:1.5b")
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.port, 8765)
        self.assertEqual(cfg.root, ROOT)

    def test_placeholder_warning_names_the_variable(self) -> None:
        with env_set(OLLAMA_BASE=None):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError) as caught:
                    server.config_from_env(root=Path(tmp))
        self.assertIn("OLLAMA_BASE", str(caught.exception))


def env_set(**values: str | None):
    saved = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    @contextlib.contextmanager
    def _restore():
        try:
            yield
        finally:
            for key, old in saved.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

    return _restore()


class SecurityTests(ProxyFixture, unittest.TestCase):
    def test_public_bind_is_refused(self) -> None:
        for host in ("0.0.0.0", "192.168.1.5", "100.64.0.2", "::"):
            with self.subTest(host=host):
                with env_set(OLLAMA_BASE="http://100.64.1.8:11434", CHAT_HOST=host):
                    with self.assertRaises(ValueError) as caught:
                        server.config_from_env()
                self.assertIn("127.0.0.1", str(caught.exception))

    def test_localhost_is_stored_as_loopback(self) -> None:
        with env_set(OLLAMA_BASE="http://127.0.0.1:11434", CHAT_HOST="localhost"):
            cfg = server.config_from_env()
        self.assertEqual(cfg.host, "127.0.0.1")

    def test_upstream_must_be_a_private_http_url(self) -> None:
        rejected = (
            "file:///etc/passwd",
            "gopher://127.0.0.1:70",
            "http://8.8.8.8:11434",
            "http://169.254.169.254:80",
            "http://100.128.0.1:11434",
            "http://100.x.x.x:11434",
            "http://100.64.0.2:11434/api/chat",
            "http://100.64.0.2:11434?x=1",
            "http://user:pass@127.0.0.1:11434",
            "http://evil.example:11434",
        )
        for base in rejected:
            with self.subTest(base=base):
                with env_set(OLLAMA_BASE=base):
                    with self.assertRaises(ValueError) as caught:
                        server.config_from_env()
                self.assertIn("OLLAMA_BASE", str(caught.exception))

    def test_home_and_tailscale_addresses_are_accepted(self) -> None:
        for base in (
            "http://100.64.0.2:11434",
            "http://10.1.2.3:11434",
            "http://192.168.1.20:11434",
            "http://172.16.5.5:11434",
            "https://127.0.0.1:11434",
        ):
            with self.subTest(base=base):
                with env_set(OLLAMA_BASE=base):
                    cfg = server.config_from_env()
                self.assertEqual(cfg.ollama_base, base)

    def test_local_file_sets_the_base_and_env_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.local.json"
            path.write_text(
                json.dumps({"ollama_base": "http://10.0.0.8:11434", "model": "qwen2.5:1.5b"}),
                encoding="utf-8",
            )
            with env_set(OLLAMA_BASE=None, OLLAMA_MODEL=None):
                cfg = server.config_from_env(root=Path(tmp))
            self.assertEqual(cfg.ollama_base, "http://10.0.0.8:11434")
            with env_set(OLLAMA_BASE="http://100.64.9.9:11434"):
                cfg = server.config_from_env(root=Path(tmp))
            self.assertEqual(cfg.ollama_base, "http://100.64.9.9:11434")

    def test_file_cannot_choose_the_listen_address(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.local.json"
            path.write_text(
                json.dumps(
                    {
                        "ollama_base": "http://10.0.0.8:11434",
                        "host": "0.0.0.0",
                    }
                ),
                encoding="utf-8",
            )
            with env_set(OLLAMA_BASE=None, CHAT_HOST=None):
                cfg = server.config_from_env(root=Path(tmp))
            self.assertEqual(cfg.host, "127.0.0.1")

    def test_browser_cannot_retarget_the_proxy(self) -> None:
        secret_hits = {"n": 0}

        class Secret(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                secret_hits["n"] += 1
                body = b"secret"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        secret, secret_thread = listen(Secret)
        try:
            evil = f"http://127.0.0.1:{secret.server_address[1]}/steal"
            payload = json.dumps(
                {
                    "model": "qwen2.5:1.5b",
                    "ollama_base": evil,
                    "url": evil,
                    "messages": [
                        {
                            "role": "user",
                            "content": "hi",
                            "image_url": evil,
                        }
                    ],
                    "stream": True,
                }
            ).encode()
            status, _, body = http_call(self.port, f"/api/chat?ollama_base={evil}", payload)
        finally:
            stop(secret, secret_thread)
        self.assertEqual(status, 200)
        self.assertNotIn(b"secret", body)
        self.assertEqual(secret_hits["n"], 0)
        sent = json.loads(RecordingOllama.last)
        self.assertEqual(set(sent), {"model", "messages", "stream"})
        self.assertFalse(sent["stream"])
        self.assertEqual(set(sent["messages"][0]), {"role", "content"})
        self.assertEqual(sent["messages"][0]["content"], "hi")

    def test_oversized_body_is_refused(self) -> None:
        status, _, body = http_call(
            self.port,
            "/api/chat",
            b"x" * (server.MAX_BODY + 1),
        )
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body)["error"], "body too large")

    def test_too_many_messages_are_refused(self) -> None:
        messages = [
            {"role": "user", "content": "x"} for _ in range(server.MAX_MESSAGES + 1)
        ]
        status, _, body = http_call(
            self.port,
            "/api/chat",
            json.dumps({"messages": messages}).encode(),
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"], "too many messages")

    def test_config_query_does_not_change_the_base(self) -> None:
        status, _, body = http_call(
            self.port,
            "/api/config?ollama_base=http://8.8.8.8:11434",
        )
        self.assertEqual(status, 200)
        self.assertNotIn("8.8.8.8", body.decode())

    def test_local_config_file_is_not_served(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secret = Path(tmp) / "chat.local.json"
            secret.write_text('{"ollama_base":"http://10.0.0.8:11434"}', encoding="utf-8")
            cfg = server.Config(
                ollama_base="http://127.0.0.1:9",
                model="qwen2.5:1.5b",
                host="127.0.0.1",
                port=8765,
                root=Path(tmp),
            )
            httpd, thread = listen(server.make_handler(cfg))
            try:
                status, _, body = http_call(httpd.server_address[1], "/chat.local.json")
                sneak, _, sneak_body = http_call(
                    httpd.server_address[1], "/%2e%2e/chat.local.json"
                )
            finally:
                stop(httpd, thread)
        self.assertEqual(status, 404)
        self.assertNotIn(b"10.0.0.8", body)
        self.assertEqual(sneak, 404)
        self.assertNotIn(b"10.0.0.8", sneak_body)

    def test_responses_have_no_open_cors(self) -> None:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/")
        with urllib.request.urlopen(req, timeout=5) as resp:
            allow = resp.headers.get("Access-Control-Allow-Origin")
        self.assertNotEqual(allow, "*")
        self.assertIsNone(allow)

    def test_log_does_not_include_the_prompt(self) -> None:
        secret = "session-secret-phrase"
        payload = json.dumps(
            {"messages": [{"role": "user", "content": secret}]}
        ).encode()
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            http_call(self.port, "/api/chat", payload)
        self.assertNotIn(secret, captured.getvalue())

    def test_upstream_redirect_is_not_followed(self) -> None:
        hits = {"n": 0}

        class Secret(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                hits["n"] += 1
                body = b"secret"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET

            def log_message(self, fmt: str, *args) -> None:
                return

        class Bounce(BaseHTTPRequestHandler):
            location = ""

            def do_POST(self) -> None:
                self.send_response(302)
                self.send_header("Location", Bounce.location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, fmt: str, *args) -> None:
                return

        secret, secret_thread = listen(Secret)
        Bounce.location = f"http://127.0.0.1:{secret.server_address[1]}/secret"
        bounce, bounce_thread = listen(Bounce)
        cfg = server.Config(
            ollama_base=f"http://127.0.0.1:{bounce.server_address[1]}",
            model="qwen2.5:1.5b",
            host="127.0.0.1",
            port=8765,
            root=ROOT,
        )
        proxy, proxy_thread = listen(server.make_handler(cfg))
        try:
            status, _, body = http_call(
                proxy.server_address[1],
                "/api/chat",
                b'{"messages":[{"role":"user","content":"x"}]}',
            )
        finally:
            stop(proxy, proxy_thread)
            stop(bounce, bounce_thread)
            stop(secret, secret_thread)
        self.assertNotEqual(status, 200)
        self.assertEqual(hits["n"], 0)
        self.assertNotIn(b"secret", body)

    def test_page_does_not_inject_model_html(self) -> None:
        text = (ROOT / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", text)
        self.assertNotIn("insertAdjacentHTML", text)
        self.assertNotIn("document.write", text)
        self.assertIn("textContent", text)
        self.assertIn("AbortController", text)
        self.assertIn("turnId", text)
        self.assertIn("/api/health", text)
        self.assertIn("Остановить", text)


def quiet(handler):
    def log_message(self, fmt: str, *args) -> None:
        return

    handler.log_message = log_message
    return handler


class ReliabilityTests(unittest.TestCase):
    def _proxy(self, upstream_port: int):
        cfg = server.Config(
            ollama_base=f"http://127.0.0.1:{upstream_port}",
            model="qwen2.5:1.5b",
            host="127.0.0.1",
            port=8765,
            root=ROOT,
        )
        return listen(server.make_handler(cfg))

    def test_health_splits_proxy_from_a_live_backend(self) -> None:
        class Tags(BaseHTTPRequestHandler):
            seen = {}

            def do_GET(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                Tags.seen = {
                    "path": self.path,
                    "body": self.rfile.read(length) if length else b"",
                }
                body = b'{"models":[]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        upstream, upstream_thread = listen(Tags)
        proxy, proxy_thread = self._proxy(upstream.server_address[1])
        try:
            status, _, body = http_call(proxy.server_address[1], "/api/health")
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["proxy"], "ok")
        self.assertEqual(data["backend"], "ok")
        self.assertEqual(Tags.seen.get("path"), "/api/tags")
        self.assertEqual(Tags.seen.get("body"), b"")
        self.assertNotIn(b"messages", body)

    def test_health_reports_an_offline_backend(self) -> None:
        cfg = server.Config(
            ollama_base="http://127.0.0.1:9",
            model="qwen2.5:1.5b",
            host="127.0.0.1",
            port=8765,
            root=ROOT,
        )
        proxy, proxy_thread = listen(server.make_handler(cfg))
        try:
            status, _, body = http_call(proxy.server_address[1], "/api/health")
        finally:
            stop(proxy, proxy_thread)
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["proxy"], "ok")
        self.assertEqual(data["backend"], "refused")

    def test_hanging_backend_fails_within_the_read_budget(self) -> None:
        class Hang(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                time.sleep(8)

            def log_message(self, fmt: str, *args) -> None:
                return

        upstream, upstream_thread = listen(Hang)
        proxy, proxy_thread = self._proxy(upstream.server_address[1])
        with env_set(OLLAMA_CONNECT_TIMEOUT="0.3", OLLAMA_READ_TIMEOUT="0.4"):
            started = time.monotonic()
            try:
                status, _, body = http_call(
                    proxy.server_address[1],
                    "/api/chat",
                    b'{"messages":[{"role":"user","content":"x"}]}',
                    timeout=3,
                )
            finally:
                stop(proxy, proxy_thread)
                stop(upstream, upstream_thread)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.5)
        self.assertEqual(status, 504)
        self.assertEqual(json.loads(body)["error"], "timeout")

    def test_client_disconnect_closes_the_upstream(self) -> None:
        class Hold(BaseHTTPRequestHandler):
            holding = threading.Event()
            dropped = threading.Event()
            phase = 0

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length:
                    self.rfile.read(length)
                if Hold.phase == 0:
                    Hold.phase = 1
                    Hold.holding.set()
                    self.connection.settimeout(3)
                    try:
                        chunk = self.connection.recv(1)
                    except OSError:
                        Hold.dropped.set()
                        return
                    if chunk == b"":
                        Hold.dropped.set()
                    return
                body = assistant_payload("дальше")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        Hold.holding.clear()
        Hold.dropped.clear()
        Hold.phase = 0
        upstream, upstream_thread = listen(Hold)
        proxy, proxy_thread = self._proxy(upstream.server_address[1])
        try:
            sock = socket.create_connection(("127.0.0.1", proxy.server_address[1]))
            raw = b'{"messages":[{"role":"user","content":"cancel-me"}]}'
            request = (
                b"POST /api/chat HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(raw)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + raw
            )
            sock.sendall(request)
            self.assertTrue(Hold.holding.wait(2))
            sock.close()
            self.assertTrue(Hold.dropped.wait(2))
            status, body = 429, b""
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status, _, body = http_call(
                    proxy.server_address[1],
                    "/api/chat",
                    b'{"messages":[{"role":"user","content":"next"}]}',
                )
                if status != 429:
                    break
                time.sleep(0.05)
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(status, 200)
        self.assertNotIn(b"cancel-me", body)

    def test_a_second_chat_is_busy_until_the_first_finishes(self) -> None:
        class Gate(BaseHTTPRequestHandler):
            started = threading.Event()
            release = threading.Event()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length:
                    self.rfile.read(length)
                Gate.started.set()
                Gate.release.wait(3)
                body = assistant_payload("первый")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        Gate.started.clear()
        Gate.release.clear()
        upstream, upstream_thread = listen(Gate)
        proxy, proxy_thread = self._proxy(upstream.server_address[1])
        first: dict = {}

        def run_first() -> None:
            first["result"] = http_call(
                proxy.server_address[1],
                "/api/chat",
                b'{"messages":[{"role":"user","content":"hold"}]}',
                timeout=5,
            )

        worker = threading.Thread(target=run_first)
        worker.start()
        try:
            self.assertTrue(Gate.started.wait(2))
            status, _, body = http_call(
                proxy.server_address[1],
                "/api/chat",
                b'{"messages":[{"role":"user","content":"later"}]}',
            )
            self.assertEqual(status, 429)
            self.assertEqual(json.loads(body)["error"], "busy")
            self.assertNotIn(b"later", body)
            Gate.release.set()
            worker.join(5)
        finally:
            Gate.release.set()
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(first["result"][0], 200)

    def test_request_id_is_echoed_only_when_safe(self) -> None:
        class Reply(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = assistant_payload("ок")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        upstream, upstream_thread = listen(Reply)
        proxy, proxy_thread = self._proxy(upstream.server_address[1])
        try:
            safe = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/api/chat",
                data=b'{"messages":[{"role":"user","content":"x"}]}',
                method="POST",
                headers={"Content-Type": "application/json", "X-Request-Id": "17"},
            )
            with urllib.request.urlopen(safe, timeout=5) as resp:
                echoed = resp.headers.get("X-Request-Id")
            junk = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/api/chat",
                data=b'{"messages":[{"role":"user","content":"x"}]}',
                method="POST",
                headers={"Content-Type": "application/json", "X-Request-Id": "<script>"},
            )
            with urllib.request.urlopen(junk, timeout=5) as resp:
                rejected = resp.headers.get("X-Request-Id")
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(echoed, "17")
        self.assertIsNone(rejected)


def openai_reply(text: str, reasoning: str = "") -> bytes:
    message = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning"] = reasoning
    return json.dumps({"choices": [{"message": message}]}).encode()


def lm_config(port: int, model: str = "qwen/qwen3.5-9b") -> server.Config:
    return server.Config(
        ollama_base="",
        model=model,
        host="127.0.0.1",
        port=8765,
        root=ROOT,
        provider="lmstudio",
        lmstudio_base=f"http://127.0.0.1:{port}",
    )


class RecordingStudio(BaseHTTPRequestHandler):
    status = 200
    payload = b""
    last = b""
    path = ""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        RecordingStudio.last = self.rfile.read(length) if length else b""
        RecordingStudio.path = self.path
        body = self.payload
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        RecordingStudio.path = self.path
        RecordingStudio.last = b""
        body = self.payload or b'{"data":[]}'
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return


class LmStudioTests(unittest.TestCase):
    def test_ollama_remains_the_default(self) -> None:
        with env_set(
            CHAT_PROVIDER=None,
            LMSTUDIO_BASE="http://127.0.0.1:1234",
            LMSTUDIO_MODEL="qwen/qwen3.5-9b",
            OLLAMA_BASE="http://100.64.0.2:11434",
            OLLAMA_MODEL=None,
        ):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = server.config_from_env(root=Path(tmp))
        self.assertEqual(cfg.provider, "ollama")
        self.assertEqual(cfg.ollama_base, "http://100.64.0.2:11434")
        self.assertEqual(cfg.lmstudio_base, "")
        self.assertEqual(cfg.model, "qwen2.5:1.5b")

    def test_explicit_loopback_and_model_are_required(self) -> None:
        rejected = (
            "http://100.64.0.2:1234",
            "http://10.1.2.3:1234",
            "http://8.8.8.8:1234",
            "http://127.0.0.2:1234",
            "http://127.0.0.1:1234/v1",
            "http://user:pass@127.0.0.1:1234",
            "file:///tmp/model",
        )
        for base in rejected:
            with self.subTest(base=base):
                with env_set(
                    CHAT_PROVIDER="lmstudio",
                    LMSTUDIO_BASE=base,
                    LMSTUDIO_MODEL="qwen/qwen3.5-9b",
                    OLLAMA_BASE="http://8.8.8.8:11434",
                ):
                    with tempfile.TemporaryDirectory() as tmp:
                        with self.assertRaises(ValueError) as caught:
                            server.config_from_env(root=Path(tmp))
                self.assertIn("LMSTUDIO_BASE", str(caught.exception))
                self.assertNotIn("8.8.8.8", str(caught.exception))
        with env_set(
            CHAT_PROVIDER="lmstudio",
            LMSTUDIO_BASE="http://localhost:1234/",
            LMSTUDIO_MODEL="qwen/qwen3.5-9b",
            OLLAMA_BASE=None,
        ):
            with tempfile.TemporaryDirectory() as tmp:
                cfg = server.config_from_env(root=Path(tmp))
        self.assertEqual(cfg.lmstudio_base, "http://127.0.0.1:1234")
        self.assertEqual(cfg.ollama_base, "")
        with env_set(CHAT_PROVIDER="cloud", OLLAMA_BASE="http://127.0.0.1:11434"):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError) as caught:
                    server.config_from_env(root=Path(tmp))
        self.assertIn("CHAT_PROVIDER", str(caught.exception))
        with env_set(CHAT_PROVIDER="lmstudio", LMSTUDIO_BASE="http://127.0.0.1:1234", LMSTUDIO_MODEL=None):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    server.config_from_env(root=Path(tmp))

    def test_file_sets_lmstudio_and_env_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat.local.json"
            path.write_text(
                json.dumps(
                    {
                        "provider": "lmstudio",
                        "lmstudio_base": "http://127.0.0.1:1234",
                        "model": "google/gemma-4-e4b",
                        "ollama_base": "http://10.0.0.8:11434",
                    }
                ),
                encoding="utf-8",
            )
            with env_set(CHAT_PROVIDER=None, LMSTUDIO_BASE=None, LMSTUDIO_MODEL=None, OLLAMA_BASE=None):
                cfg = server.config_from_env(root=Path(tmp))
            self.assertEqual(cfg.provider, "lmstudio")
            self.assertEqual(cfg.model, "google/gemma-4-e4b")
            self.assertEqual(cfg.lmstudio_base, "http://127.0.0.1:1234")
            with env_set(
                CHAT_PROVIDER="lmstudio",
                LMSTUDIO_BASE="http://127.0.0.1:1235",
                LMSTUDIO_MODEL="qwen/qwen3.5-9b",
            ):
                cfg = server.config_from_env(root=Path(tmp))
        self.assertEqual(cfg.lmstudio_base, "http://127.0.0.1:1235")
        self.assertEqual(cfg.model, "qwen/qwen3.5-9b")

    def test_chat_is_openai_compatible_and_hides_reasoning(self) -> None:
        RecordingStudio.status = 200
        RecordingStudio.payload = openai_reply("привет", "scratch")
        RecordingStudio.last = b""
        upstream, upstream_thread = listen(RecordingStudio)
        proxy, proxy_thread = listen(server.make_handler(lm_config(upstream.server_address[1])))
        try:
            payload = json.dumps(
                {
                    "model": "other/model",
                    "lmstudio_base": "http://127.0.0.1:9",
                    "tools": [{"type": "function", "function": {"name": "mcp"}}],
                    "messages": [
                        {
                            "role": "user",
                            "content": "ты здесь?",
                            "image_url": "http://127.0.0.1:9/secret.png",
                        }
                    ],
                }
            ).encode()
            status, _, body = http_call(proxy.server_address[1], "/api/chat?lmstudio_base=http://8.8.8.8:1", payload)
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["message"]["content"], "привет")
        self.assertNotIn(b"scratch", body)
        self.assertNotIn(b"8.8.8.8", body)
        sent = json.loads(RecordingStudio.last)
        self.assertEqual(RecordingStudio.path, "/v1/chat/completions")
        self.assertEqual(
            set(sent),
            {"model", "messages", "stream", "max_tokens", "reasoning_effort"},
        )
        self.assertEqual(sent["model"], "qwen/qwen3.5-9b")
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["max_tokens"], server.LMSTUDIO_MAX_TOKENS)
        self.assertEqual(sent["reasoning_effort"], "none")
        self.assertEqual(set(sent["messages"][0]), {"role", "content"})
        self.assertEqual(sent["messages"][0]["content"], "ты здесь?")

    def test_reasoning_only_reply_is_not_shown(self) -> None:
        RecordingStudio.status = 200
        RecordingStudio.payload = openai_reply("  ", "hidden-chain")
        upstream, upstream_thread = listen(RecordingStudio)
        proxy, proxy_thread = listen(server.make_handler(lm_config(upstream.server_address[1])))
        try:
            status, _, body = http_call(
                proxy.server_address[1],
                "/api/chat",
                b'{"messages":[{"role":"user","content":"x"}]}',
            )
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(status, 502)
        self.assertEqual(json.loads(body)["error"], "unreachable")
        self.assertNotIn(b"hidden-chain", body)

    def test_upstream_error_does_not_echo_the_prompt(self) -> None:
        secret = "session-secret-phrase"
        RecordingStudio.status = 500
        RecordingStudio.payload = secret.encode()
        upstream, upstream_thread = listen(RecordingStudio)
        proxy, proxy_thread = listen(server.make_handler(lm_config(upstream.server_address[1])))
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                status, _, body = http_call(
                    proxy.server_address[1],
                    "/api/chat",
                    json.dumps({"messages": [{"role": "user", "content": secret}]}).encode(),
                )
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"], "unreachable")
        self.assertNotIn(secret.encode(), body)
        self.assertNotIn(secret, captured.getvalue())

    def test_health_checks_models_not_tags(self) -> None:
        RecordingStudio.status = 200
        RecordingStudio.payload = b'{"data":[{"id":"qwen/qwen3.5-9b"}]}'
        RecordingStudio.path = ""
        upstream, upstream_thread = listen(RecordingStudio)
        proxy, proxy_thread = listen(server.make_handler(lm_config(upstream.server_address[1])))
        try:
            status, _, body = http_call(proxy.server_address[1], "/api/health")
            cfg_status, _, cfg_body = http_call(
                proxy.server_address[1],
                "/api/config?provider=ollama&lmstudio_base=http://8.8.8.8:9",
            )
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["backend"], "ok")
        self.assertEqual(RecordingStudio.path, "/v1/models")
        self.assertNotIn(b"messages", body)
        self.assertEqual(cfg_status, 200)
        reported = json.loads(cfg_body)
        self.assertEqual(reported["provider"], "lmstudio")
        self.assertEqual(reported["model"], "qwen/qwen3.5-9b")
        self.assertTrue(reported["upstream"].startswith("http://127.0.0.1:"))
        self.assertNotIn("8.8.8.8", cfg_body.decode())

    def test_offline_health_stays_a_proxy_ok(self) -> None:
        proxy, proxy_thread = listen(server.make_handler(lm_config(9)))
        try:
            status, _, body = http_call(proxy.server_address[1], "/api/health")
        finally:
            stop(proxy, proxy_thread)
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["proxy"], "ok")
        self.assertEqual(data["backend"], "refused")

    def test_redirect_is_not_followed(self) -> None:
        hits = {"n": 0}

        class Secret(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                hits["n"] += 1
                body = b"secret"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        class Bounce(BaseHTTPRequestHandler):
            location = ""

            def do_POST(self) -> None:
                self.send_response(302)
                self.send_header("Location", Bounce.location)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, fmt: str, *args) -> None:
                return

        secret, secret_thread = listen(Secret)
        Bounce.location = f"http://127.0.0.1:{secret.server_address[1]}/secret"
        bounce, bounce_thread = listen(Bounce)
        proxy, proxy_thread = listen(server.make_handler(lm_config(bounce.server_address[1])))
        try:
            status, _, body = http_call(
                proxy.server_address[1],
                "/api/chat",
                b'{"messages":[{"role":"user","content":"x"}]}',
            )
        finally:
            stop(proxy, proxy_thread)
            stop(bounce, bounce_thread)
            stop(secret, secret_thread)
        self.assertEqual(status, 502)
        self.assertEqual(hits["n"], 0)
        self.assertNotIn(b"secret", body)

    def test_timeout_busy_and_cancel_keep_one_slot(self) -> None:
        class Hang(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                time.sleep(8)

            def log_message(self, fmt: str, *args) -> None:
                return

        upstream, upstream_thread = listen(Hang)
        proxy, proxy_thread = listen(server.make_handler(lm_config(upstream.server_address[1])))
        with env_set(OLLAMA_CONNECT_TIMEOUT="0.3", OLLAMA_READ_TIMEOUT="0.4"):
            started = time.monotonic()
            try:
                status, _, body = http_call(
                    proxy.server_address[1],
                    "/api/chat",
                    b'{"messages":[{"role":"user","content":"x"}]}',
                    timeout=3,
                )
            finally:
                stop(proxy, proxy_thread)
                stop(upstream, upstream_thread)
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertEqual(status, 504)
        self.assertEqual(json.loads(body)["error"], "timeout")

        class Gate(BaseHTTPRequestHandler):
            started = threading.Event()
            release = threading.Event()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length:
                    self.rfile.read(length)
                Gate.started.set()
                Gate.release.wait(3)
                body = openai_reply("первый")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        Gate.started.clear()
        Gate.release.clear()
        upstream, upstream_thread = listen(Gate)
        proxy, proxy_thread = listen(server.make_handler(lm_config(upstream.server_address[1])))
        first: dict = {}

        def run_first() -> None:
            first["result"] = http_call(
                proxy.server_address[1],
                "/api/chat",
                b'{"messages":[{"role":"user","content":"hold"}]}',
                timeout=5,
            )

        worker = threading.Thread(target=run_first)
        worker.start()
        try:
            self.assertTrue(Gate.started.wait(2))
            status, _, body = http_call(
                proxy.server_address[1],
                "/api/chat",
                b'{"messages":[{"role":"user","content":"later"}]}',
            )
            self.assertEqual(status, 429)
            self.assertEqual(json.loads(body)["error"], "busy")
            Gate.release.set()
            worker.join(5)
        finally:
            Gate.release.set()
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(first["result"][0], 200)

        class Hold(BaseHTTPRequestHandler):
            holding = threading.Event()
            dropped = threading.Event()
            phase = 0

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or 0)
                if length:
                    self.rfile.read(length)
                if Hold.phase == 0:
                    Hold.phase = 1
                    Hold.holding.set()
                    self.connection.settimeout(3)
                    try:
                        chunk = self.connection.recv(1)
                    except OSError:
                        Hold.dropped.set()
                        return
                    if chunk == b"":
                        Hold.dropped.set()
                    return
                body = openai_reply("дальше")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        Hold.holding.clear()
        Hold.dropped.clear()
        Hold.phase = 0
        upstream, upstream_thread = listen(Hold)
        proxy, proxy_thread = listen(server.make_handler(lm_config(upstream.server_address[1])))
        try:
            sock = socket.create_connection(("127.0.0.1", proxy.server_address[1]))
            raw = b'{"messages":[{"role":"user","content":"cancel-me"}]}'
            request = (
                b"POST /api/chat HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(raw)).encode() + b"\r\n"
                b"Connection: close\r\n\r\n" + raw
            )
            sock.sendall(request)
            self.assertTrue(Hold.holding.wait(2))
            sock.close()
            self.assertTrue(Hold.dropped.wait(2))
            req = urllib.request.Request(
                f"http://127.0.0.1:{proxy.server_address[1]}/api/chat",
                data=b'{"messages":[{"role":"user","content":"next"}]}',
                method="POST",
                headers={"Content-Type": "application/json", "X-Request-Id": "17"},
            )
            echoed = None
            status = 429
            body = b""
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        status = resp.status
                        echoed = resp.headers.get("X-Request-Id")
                        body = resp.read()
                except urllib.error.HTTPError as err:
                    status = err.code
                    body = err.read()
                if status != 429:
                    break
                time.sleep(0.05)
        finally:
            stop(proxy, proxy_thread)
            stop(upstream, upstream_thread)
        self.assertEqual(status, 200)
        self.assertEqual(echoed, "17")
        self.assertNotIn(b"cancel-me", body)
        self.assertEqual(json.loads(body)["message"]["content"], "дальше")


if __name__ == "__main__":
    unittest.main()
