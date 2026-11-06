#!/usr/bin/env python3
"""Local chat proxy for a browser on this Mac.

Glossary:
- Proxy: this process. The browser talks only to it.
- Upstream: one fixed base, from the environment or chat.local.json.
- Provider: ollama (default) or an explicit loopback LM Studio.
- Turn: one chat message, a role and some content.
- Transcript: the turns the page sends as messages.

The proxy listens on 127.0.0.1 only. The browser cannot choose the upstream.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import re
import select
import socket
import sys
import threading
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG = "chat.local.json"
DEFAULT_MODEL = "qwen2.5:1.5b"
MAX_BODY = 1_500_000
MAX_MESSAGES = 40
MAX_CONTENT = 32_000
JSON = "application/json; charset=utf-8"
TEXT = "text/plain; charset=utf-8"
BASE_ERROR = (
    "Set OLLAMA_BASE to http://100.64.x.x:11434, a home LAN address, "
    "or 127.0.0.1 with a port and no path. "
    "A chat.local.json file beside server.py may set ollama_base instead. "
    "OLLAMA_BASE is rejected when it is public, a file URL, or has a path."
)
HOST_ERROR = "CHAT_HOST must be 127.0.0.1. The chat proxy does not listen on the network."
MODEL_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}(?:/[A-Za-z0-9][A-Za-z0-9_.:-]{0,79})?$"
)
LMSTUDIO_MAX_TOKENS = 512
# LM Studio 0.4.8 accepts reasoning_effort on /v1/chat/completions.
# "none" is OpenAI's documented no-reasoning value. Native reasoning "off"
# belongs to /api/v1/chat and is not sent on this path.
LMSTUDIO_REASONING = "none"
PROVIDER_ERROR = "CHAT_PROVIDER must be ollama or lmstudio."
LMSTUDIO_ERROR = (
    "Set CHAT_PROVIDER=lmstudio, LMSTUDIO_MODEL to a model id, and "
    "LMSTUDIO_BASE to http://127.0.0.1:1234 with a port and no path. "
    "chat.local.json may set provider, lmstudio_base, and model instead. "
    "Other hosts are rejected. OLLAMA_BASE is not used."
)
ROLES = frozenset({"system", "user", "assistant"})
ALLOWED = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
)

PAGES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/tokens.css": ("tokens.css", "text/css; charset=utf-8"),
    "/base.css": ("base.css", "text/css; charset=utf-8"),
}
CHAT_SLOT = threading.BoundedSemaphore(1)
BACKEND_CODES = frozenset({"timeout", "refused", "name", "unreachable"})


@dataclass(frozen=True)
class Config:
    ollama_base: str
    model: str
    host: str
    port: int
    root: Path
    provider: str = "ollama"
    lmstudio_base: str = ""

    def __post_init__(self) -> None:
        provider = clean_provider(self.provider)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "host", loopback_host(self.host))
        object.__setattr__(self, "port", tcp_port(self.port))
        if provider == "lmstudio":
            chosen = _text(self.model)
            if not chosen:
                raise ValueError(LMSTUDIO_ERROR)
            object.__setattr__(self, "model", clean_model(chosen, chosen))
            object.__setattr__(self, "lmstudio_base", loopback_base(self.lmstudio_base))
            object.__setattr__(self, "ollama_base", "")
        else:
            object.__setattr__(self, "model", clean_model(self.model, DEFAULT_MODEL))
            object.__setattr__(self, "ollama_base", normalize_base(self.ollama_base))
            object.__setattr__(self, "lmstudio_base", "")


def loopback_host(host: str) -> str:
    cleaned = host.strip().lower()
    if cleaned == "localhost" or cleaned == "127.0.0.1":
        return "127.0.0.1"
    raise ValueError(HOST_ERROR)


def tcp_port(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("CHAT_PORT must be a number")
    try:
        port = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError("CHAT_PORT must be a number") from None
    if port < 1 or port > 65535:
        raise ValueError("CHAT_PORT must be a number")
    return port


def host_is_allowed(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return any(address in network for network in ALLOWED)


def reject_decorations(raw: str, parts: urllib.parse.SplitResult) -> None:
    if "\\" in raw or "@" in raw or parts.username or parts.password:
        raise ValueError(BASE_ERROR)
    if parts.query or parts.fragment or parts.path not in {"", "/"}:
        raise ValueError(BASE_ERROR)


def clean_provider(value: object) -> str:
    if value is None:
        return "ollama"
    if not isinstance(value, str):
        raise ValueError(PROVIDER_ERROR)
    cleaned = value.strip().lower()
    if cleaned in {"", "ollama"}:
        return "ollama"
    if cleaned == "lmstudio":
        return "lmstudio"
    raise ValueError(PROVIDER_ERROR)


def loopback_base(url: str) -> str:
    try:
        normalized = normalize_base(url)
    except ValueError:
        raise ValueError(LMSTUDIO_ERROR) from None
    parts = urllib.parse.urlsplit(normalized)
    host = parts.hostname or ""
    if host == "localhost":
        return f"{parts.scheme}://127.0.0.1:{parts.port}"
    if host != "127.0.0.1":
        raise ValueError(LMSTUDIO_ERROR)
    return normalized


def active_base(config: Config) -> str:
    if config.provider == "lmstudio":
        return config.lmstudio_base
    return config.ollama_base


def normalize_base(url: str) -> str:
    raw = url.strip()
    parts = urllib.parse.urlsplit(raw)
    reject_decorations(raw, parts)
    if parts.scheme.lower() not in {"http", "https"}:
        raise ValueError(BASE_ERROR)
    host = parts.hostname
    try:
        port = parts.port
    except ValueError:
        raise ValueError(BASE_ERROR) from None
    if host is None or port is None or not host_is_allowed(host):
        raise ValueError(BASE_ERROR)
    shown = f"[{host}]" if ":" in host else host
    return f"{parts.scheme.lower()}://{shown}:{port}"


def local_settings(root: Path) -> dict:
    path = root / LOCAL_CONFIG
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("chat.local.json is not valid json") from None
    if not isinstance(data, dict):
        raise ValueError("chat.local.json is not valid json")
    return data


def _text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def config_from_env(root: Path = ROOT) -> Config:
    stored = local_settings(root)
    provider = clean_provider(
        _text(os.environ.get("CHAT_PROVIDER")) or _text(stored.get("provider")) or "ollama"
    )
    host = os.environ.get("CHAT_HOST", "127.0.0.1")
    port = tcp_port(os.environ.get("CHAT_PORT", "8765"))
    if provider == "lmstudio":
        base = _text(os.environ.get("LMSTUDIO_BASE")) or _text(stored.get("lmstudio_base"))
        model = _text(os.environ.get("LMSTUDIO_MODEL")) or _text(stored.get("model"))
        if not base or not model:
            raise ValueError(LMSTUDIO_ERROR)
        return Config(
            ollama_base="",
            model=model,
            host=host,
            port=port,
            root=root,
            provider="lmstudio",
            lmstudio_base=base,
        )
    base = _text(os.environ.get("OLLAMA_BASE")) or _text(stored.get("ollama_base"))
    if not base:
        raise ValueError(BASE_ERROR)
    model = _text(os.environ.get("OLLAMA_MODEL")) or _text(stored.get("model")) or DEFAULT_MODEL
    return Config(
        ollama_base=base,
        model=model,
        host=host,
        port=port,
        root=root,
    )


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def error_bytes(message: str) -> bytes:
    return json_bytes({"error": message})


def read_body(stream, header: str | None) -> bytes:
    if header is None or header.strip() == "":
        return b""
    try:
        length = int(header)
    except ValueError:
        raise ValueError("bad length") from None
    if length < 0:
        raise ValueError("bad length")
    chunk = stream.read(min(length, MAX_BODY + 1))
    if length > MAX_BODY or len(chunk) > MAX_BODY:
        raise ValueError("body too large")
    return chunk


def parse_object(raw: bytes) -> dict:
    if not raw:
        raise ValueError("messages required")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("bad json") from None
    if not isinstance(data, dict):
        raise ValueError("bad json")
    return data


def clean_model(value: object, default: str) -> str:
    if value is None or value == "":
        chosen = default
    elif isinstance(value, str):
        chosen = value.strip()
    else:
        raise ValueError("bad model")
    if MODEL_RE.fullmatch(chosen) is None:
        raise ValueError("bad model")
    return chosen


def clean_message(item: object) -> dict:
    if not isinstance(item, dict):
        raise ValueError("bad message")
    role = item.get("role")
    content = item.get("content")
    if role not in ROLES or not isinstance(content, str):
        raise ValueError("bad message")
    text = content.strip()
    if not text or len(text) > MAX_CONTENT:
        raise ValueError("bad message")
    return {"role": role, "content": text}


def clean_messages(items: object) -> list:
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("messages required")
    if len(items) > MAX_MESSAGES:
        raise ValueError("too many messages")
    return [clean_message(item) for item in items]


def chat_body(raw: bytes, default_model: str, provider: str = "ollama") -> bytes:
    data = parse_object(raw)
    messages = clean_messages(data.get("messages"))
    if provider == "lmstudio":
        model = clean_model(default_model, default_model)
        return json_bytes(
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": LMSTUDIO_MAX_TOKENS,
                "reasoning_effort": LMSTUDIO_REASONING,
            }
        )
    model = clean_model(data.get("model"), default_model)
    return json_bytes({"model": model, "messages": messages, "stream": False})


def read_page(root: Path, name: str) -> bytes | None:
    if "/" in name or "\\" in name or name != Path(name).name:
        return None
    path = (root / name).resolve()
    if path.parent != root.resolve() or not path.is_file():
        return None
    return path.read_bytes()


def failure_code(err: OSError) -> str:
    text = str(err).lower()
    if isinstance(err, TimeoutError) or "timed out" in text:
        return "timeout"
    if "connection refused" in text or "errno 111" in text:
        return "refused"
    if "name or service" in text or "errno -2" in text or "nodename" in text:
        return "name"
    return "unreachable"


def timeout_seconds(name: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value < low or value > high:
        return default
    return value


def connect_timeout() -> float:
    return timeout_seconds("OLLAMA_CONNECT_TIMEOUT", 5.0, 0.2, 30.0)


def read_timeout() -> float:
    return timeout_seconds("OLLAMA_READ_TIMEOUT", 120.0, 0.2, 300.0)


def health_timeout() -> float:
    return timeout_seconds("OLLAMA_HEALTH_TIMEOUT", 3.0, 0.2, 10.0)


def request_id(header: str | None) -> str | None:
    if header and re.fullmatch(r"[0-9]{1,20}", header.strip()):
        return header.strip()
    return None


def client_closed(conn: socket.socket) -> bool:
    try:
        readable, _, _ = select.select([conn], [], [], 0)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        peeked = conn.recv(1, socket.MSG_PEEK)
    except BlockingIOError:
        return False
    except OSError:
        return True
    return peeked == b""


class UpstreamCall:
    """One HTTP call to the fixed Ollama base. close() aborts it from another thread."""

    def __init__(self, base: str) -> None:
        self.base = base
        self._abort = threading.Event()
        self._lock = threading.Lock()
        self._conn: http.client.HTTPConnection | None = None

    def close(self) -> None:
        self._abort.set()
        with self._lock:
            conn = self._conn
        if conn is None:
            return
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            conn.close()
        except OSError:
            pass

    def _connect(self, connect_limit: float | None) -> http.client.HTTPConnection | None:
        if self._abort.is_set():
            return None
        parts = urllib.parse.urlsplit(self.base)
        host = parts.hostname or ""
        port = parts.port or 0
        limit = connect_timeout() if connect_limit is None else connect_limit
        if parts.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(host, port, timeout=limit)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=limit)
        with self._lock:
            if self._abort.is_set():
                return None
            self._conn = conn
        return conn

    def exchange(
        self,
        method: str,
        path: str,
        payload: bytes | None,
        read_limit: float,
        connect_limit: float | None = None,
    ) -> tuple[int, bytes]:
        conn = self._connect(connect_limit)
        if conn is None:
            return 502, error_bytes("unreachable")
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        try:
            conn.request(method, path, body=payload, headers=headers)
            if conn.sock is not None:
                conn.sock.settimeout(read_limit)
            return limited_response(conn.getresponse())
        except TimeoutError:
            return 504, error_bytes("timeout")
        except http.client.HTTPException:
            return 502, error_bytes("unreachable")
        except OSError as err:
            return transport_error(err)


def limited_response(resp: http.client.HTTPResponse) -> tuple[int, bytes]:
    if 300 <= resp.status < 400:
        return 502, error_bytes("unreachable")
    data = resp.read(MAX_BODY + 1)
    if len(data) > MAX_BODY:
        return 502, error_bytes("unreachable")
    if resp.status >= 400 and not data:
        return resp.status, error_bytes("unreachable")
    return resp.status, data


def transport_error(err: OSError) -> tuple[int, bytes]:
    code = failure_code(err)
    status = 504 if code == "timeout" else 502
    return status, error_bytes(code)


def lmstudio_reply(status: int, body: bytes) -> tuple[int, bytes]:
    if status != 200:
        code = None
        try:
            parsed = json.loads(body).get("error")
        except (UnicodeError, json.JSONDecodeError, AttributeError):
            parsed = None
        if parsed in BACKEND_CODES:
            code = str(parsed)
        return status, error_bytes(code or "unreachable")
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        return 502, error_bytes("unreachable")
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return 502, error_bytes("unreachable")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return 502, error_bytes("unreachable")
    return 200, json_bytes({"message": {"role": "assistant", "content": content}})


def probe_backend(base: str, path: str = "/api/tags") -> str:
    call = UpstreamCall(base)
    try:
        status, body = call.exchange("GET", path, None, health_timeout(), health_timeout())
    finally:
        call.close()
    if status == 200:
        return "ok"
    try:
        code = json.loads(body).get("error")
    except (json.JSONDecodeError, UnicodeError, AttributeError):
        return "unreachable"
    if code in BACKEND_CODES:
        return str(code)
    return "unreachable"


def run_chat(
    base: str,
    payload: bytes,
    connection: socket.socket,
    path: str = "/api/chat",
    adapt=None,
) -> tuple[int, bytes] | None:
    if not CHAT_SLOT.acquire(blocking=False):
        return 429, error_bytes("busy")
    call = UpstreamCall(base)
    done = threading.Event()
    box: dict[str, tuple[int, bytes]] = {}

    def worker() -> None:
        try:
            status, body = call.exchange("POST", path, payload, read_timeout())
            box["result"] = adapt(status, body) if adapt is not None else (status, body)
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    try:
        while not done.wait(0.2):
            if client_closed(connection):
                call.close()
                done.wait(2)
                return None
        if client_closed(connection):
            call.close()
            return None
        return box.get("result", (502, error_bytes("unreachable")))
    finally:
        call.close()
        CHAT_SLOT.release()


def make_handler(config: Config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            path = self.path.split("?", 1)[0]
            print(f"[{self.log_date_time_string()}] {self.command} {path}")

        def _send(self, code: int, body: bytes, content_type: str, rid: str | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if rid:
                self.send_header("X-Request-Id", rid)
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                return

        def _missing(self) -> None:
            self._send(404, b"Not found", TEXT)

        def _reject(self, err: ValueError, rid: str | None = None) -> None:
            code = 413 if str(err) == "body too large" else 400
            self._send(code, error_bytes(str(err)), JSON, rid)

        def do_GET(self) -> None:
            route = self.path.split("?", 1)[0]
            if route == "/api/config":
                body = json_bytes(
                    {
                        "provider": config.provider,
                        "ollama_base": config.ollama_base,
                        "upstream": active_base(config),
                        "model": config.model,
                    }
                )
                self._send(200, body, JSON)
                return
            if route == "/api/health":
                health_path = "/v1/models" if config.provider == "lmstudio" else "/api/tags"
                body = json_bytes(
                    {
                        "proxy": "ok",
                        "backend": probe_backend(active_base(config), health_path),
                    }
                )
                self._send(200, body, JSON)
                return
            page = PAGES.get(route)
            if page is None:
                self._missing()
                return
            body = read_page(config.root, page[0])
            if body is None:
                self._missing()
                return
            self._send(200, body, page[1])

        def do_POST(self) -> None:
            route = self.path.split("?", 1)[0]
            rid = request_id(self.headers.get("X-Request-Id"))
            if route != "/api/chat":
                self._missing()
                return
            try:
                raw = read_body(self.rfile, self.headers.get("Content-Length"))
                payload = chat_body(raw, config.model, config.provider)
            except ValueError as err:
                self._reject(err, rid)
                return
            if config.provider == "lmstudio":
                outcome = run_chat(
                    config.lmstudio_base,
                    payload,
                    self.connection,
                    "/v1/chat/completions",
                    lmstudio_reply,
                )
            else:
                outcome = run_chat(config.ollama_base, payload, self.connection)
            if outcome is None:
                return
            status, body = outcome
            self._send(status, body, JSON, rid)

    return Handler


def main() -> None:
    try:
        config = config_from_env()
    except ValueError as err:
        print(str(err), file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Chat UI → http://{config.host}:{config.port}")
    print(f"Proxy  → {config.provider} {active_base(config)}  model={config.model}")
    httpd = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    httpd.serve_forever()


if __name__ == "__main__":
    main()
