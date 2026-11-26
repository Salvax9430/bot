from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class ApiBridge:
    """Sends HTTP/REST requests, measures latency, and checks status/JSON fields."""

    def __init__(
        self,
        default_timeout_sec: float = 15.0,
        allowed_hosts: set[str] | None = None,
        max_body_chars: int = 6000,
    ) -> None:
        self.default_timeout_sec = max(1.0, float(default_timeout_sec))
        # If allowed_hosts is None, any valid http/https host is permitted;
        # if provided (e.g. {"localhost", "127.0.0.1", "api.example.com"}), only those hosts are allowed.
        self.allowed_hosts = {h.lower() for h in allowed_hosts} if allowed_hosts is not None else None
        self.max_body_chars = max(500, int(max_body_chars))

    def _check_url(self, url: str) -> str | None:
        parsed = urllib.parse.urlparse(url.strip())
        if parsed.scheme not in {"http", "https"}:
            return f"Unsupported URL scheme '{parsed.scheme}'; only http and https are allowed."
        host = (parsed.hostname or "").lower()
        if not host:
            return "URL is missing a valid hostname."
        if self.allowed_hosts is not None and host not in self.allowed_hosts:
            return f"Host '{host}' is not in allowed_hosts."
        return None

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        raw_body: str | None = None,
        expected_statuses: list[int] | None = None,
        required_json_keys: list[str] | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request and return status, latency, headers, and parsed JSON if applicable."""
        err = self._check_url(url)
        if err:
            return {"ok": False, "error": err}

        http_method = method.strip().upper() or "GET"
        req_headers: dict[str, str] = {"User-Agent": "AutonomousHumanAgent/1.0"}
        if headers:
            for k, v in headers.items():
                req_headers[str(k)] = str(v)

        data_bytes: bytes | None = None
        if json_body is not None:
            data_bytes = json.dumps(json_body, default=str).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        elif raw_body is not None:
            data_bytes = raw_body.encode("utf-8")

        req = urllib.request.Request(
            url=url.strip(),
            data=data_bytes,
            headers=req_headers,
            method=http_method,
        )

        effective_timeout = float(timeout_sec) if timeout_sec is not None else self.default_timeout_sec
        started = time.perf_counter()

        try:
            with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
                status = int(resp.getcode())
                raw_bytes = resp.read()
                text_body = raw_bytes.decode("utf-8", errors="replace")
                resp_headers = {k: v for k, v in resp.headers.items()}
                return self._evaluate_response(
                    method=http_method,
                    url=url,
                    status=status,
                    latency_ms=latency_ms,
                    headers=resp_headers,
                    text_body=text_body,
                    expected_statuses=expected_statuses,
                    required_json_keys=required_json_keys,
                )
        except urllib.error.HTTPError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            status = int(exc.code)
            err_bytes = exc.read() if exc.fp is not None else b""
            text_body = err_bytes.decode("utf-8", errors="replace")
            return self._evaluate_response(
                method=http_method,
                url=url,
                status=status,
                latency_ms=latency_ms,
                headers=dict(exc.headers.items()) if exc.headers else {},
                text_body=text_body,
                expected_statuses=expected_statuses,
                required_json_keys=required_json_keys,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return {
                "ok": False,
                "method": http_method,
                "url": url,
                "latency_ms": latency_ms,
                "error": str(exc),
            }

    def _evaluate_response(
        self,
        method: str,
        url: str,
        status: int,
        latency_ms: float,
        headers: dict[str, str],
        text_body: str,
        expected_statuses: list[int] | None,
        required_json_keys: list[str] | None,
    ) -> dict[str, Any]:
        allowed_codes = set(expected_statuses) if expected_statuses else {200, 201, 202, 204}
        status_ok = status in allowed_codes

        parsed_json: Any = None
        is_json = False
        try:
            parsed_json = json.loads(text_body)
            is_json = True
        except ValueError:
            parsed_json = None

        missing_keys: list[str] = []
        if required_json_keys:
            if not isinstance(parsed_json, dict):
                missing_keys = list(required_json_keys)
            else:
                missing_keys = [k for k in required_json_keys if k not in parsed_json]

        ok = status_ok and (len(missing_keys) == 0)
        body_preview = text_body[-self.max_body_chars :] if len(text_body) > self.max_body_chars else text_body

        return {
            "ok": ok,
            "method": method,
            "url": url,
            "status_code": status,
            "status_ok": status_ok,
            "latency_ms": latency_ms,
            "is_json": is_json,
            "json_data": parsed_json if is_json else None,
            "missing_json_keys": missing_keys,
            "headers": headers,
            "body_preview": body_preview,
        }

    def as_tool_map(self) -> dict[str, Any]:
        async def _http_request(args: dict[str, Any]) -> dict[str, Any]:
            return self.request(
                method=str(args.get("method", "GET")),
                url=str(args.get("url", "")),
                headers=args.get("headers") if isinstance(args.get("headers"), dict) else None,
                json_body=args.get("json_body"),
                raw_body=args.get("raw_body") if isinstance(args.get("raw_body"), str) else None,
                expected_statuses=args.get("expected_statuses")
                if isinstance(args.get("expected_statuses"), list)
                else None,
                required_json_keys=args.get("required_json_keys")
                if isinstance(args.get("required_json_keys"), list)
                else None,
                timeout_sec=float(args["timeout_sec"]) if args.get("timeout_sec") is not None else None,
            )

        return {
            "http_request": _http_request,
        }