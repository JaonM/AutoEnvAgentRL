"""Loopback-only HTTP client for driving an isolated sandbox runtime."""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


class HTTPSandboxClient:
    """Expose the sandbox ``handle`` protocol over a safe local HTTP origin.

    Credentials are supplied as request headers by callers and are never
    accepted as part of the URL.  Proxy discovery is disabled so loopback
    evidence cannot accidentally leave the host through a configured proxy.
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "sandbox base URL must be a credential-free loopback HTTP origin"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def handle(
        self,
        method: str,
        path: str,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, Any, Mapping[str, str]]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request_headers = {"Accept": "application/json", **dict(headers or {})}
        if payload is not None:
            request_headers["Content-Type"] = "application/json"
        request = Request(
            self.base_url + path,
            data=payload,
            headers=request_headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
                return (
                    response.status,
                    json.loads(raw) if raw else {},
                    dict(response.headers.items()),
                )
        except HTTPError as exc:
            raw = exc.read()
            try:
                value = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                value = {"error": {"code": "INVALID_HTTP_RESPONSE"}}
            return exc.code, value, dict(exc.headers.items())
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("sandbox loopback HTTP request failed") from exc


def evaluator_http_timeout(timeout_seconds: float, max_retries: int) -> float:
    """Allow the container to finish a bounded evaluator call and its retries."""
    return max(30.0, timeout_seconds * (max_retries + 1) + 15.0)
