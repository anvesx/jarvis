"""Who may use the agent service.

The service has no accounts, and what it can do costs money and shows conversations: it starts LLM and Jev work, calls
the speech engines (Cartesia bills for audio) and lists every session. So it is private: only this computer is served.

A **remote request** is one that comes through a tunnel or from another device. It has a Host that is not a loopback
name, or a header that a proxy or tunnel adds (Cloudflare's quick tunnel adds CF-Connecting-IP, CF-Ray and
X-Forwarded-For; the edge sets them, so a client cannot remove them). A remote request is served only with the key
ORB_SHARE_KEY, at least 16 characters. Without that setting, every remote request is refused.

The key goes into the link once (`https://host/?key=...`). The service then sets a cookie and redirects to the same
address without the key, so the key stays out of the address bar, the history and the Referer header. A script can
send the key in the header X-Orb-Key, or as a Bearer token.

Limit: the test is made of the Host header and the proxy headers. A proxy that hides both (it rewrites Host to
localhost and adds no forwarding header) looks like this computer. Use a tunnel that keeps them (cloudflared, ngrok).
"""
from __future__ import annotations

import hmac
import html
import logging
import time
from urllib.parse import urlencode

from starlette.requests import HTTPConnection
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

log = logging.getLogger("orb.agent")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
FORWARDED = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-forwarded-host", "x-real-ip", "true-client-ip", "forwarded")
COOKIE = "orb_key"
MIN_KEY = 16
COOKIE_DAYS = 30
LOG_EVERY = 10.0  # seconds: a scanner can send thousands of requests, so the log gets one line for each period
MESSAGE = "This orb is private. Open the link, with its key, that the owner gave you."


def host_of(request: HTTPConnection) -> str:
    """The host name of the request, without the port and the brackets of an IPv6 address."""
    host = request.headers.get("host", "").strip().lower()
    if host.startswith("["):  # [::1]:8100
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def is_remote(request: HTTPConnection) -> bool:
    return host_of(request) not in LOOPBACK or any(h in request.headers for h in FORWARDED)


def is_https(request: HTTPConnection) -> bool:
    return (request.headers.get("x-forwarded-proto", "").lower() == "https" or '"https"' in request.headers.get("cf-visitor", "")
            or request.url.scheme == "https")


def given_key(request: HTTPConnection) -> str:
    auth = request.headers.get("authorization", "")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    return request.cookies.get(COOKIE) or request.headers.get("x-orb-key") or bearer or ""


def same(a: str, b: str) -> bool:
    return bool(a) and hmac.compare_digest(a.encode(), b.encode())


def check_key(key: str | None) -> str | None:
    """The key to use, or None for no sharing. A short key is refused: a tunnel puts the service on the internet."""
    key = (key or "").strip()
    if key and len(key) < MIN_KEY:
        raise ValueError(f"ORB_SHARE_KEY must have {MIN_KEY} or more characters. Make one with: openssl rand -hex 16")
    return key or None


def denied(request: HTTPConnection) -> Response:
    if "text/html" in request.headers.get("accept", "") and request.scope.get("method") == "GET":
        return HTMLResponse(f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>orb is private</title><style>body{{font:16px/1.5 system-ui,sans-serif;display:grid;place-items:center;min-height:100vh;margin:0;
background:#0F1115;color:#E8EAEE}}p{{max-width:34ch;text-align:center}}</style><p>{html.escape(MESSAGE)}</p>""", status_code=403)
    return JSONResponse({"error": MESSAGE}, status_code=403)


class AccessMiddleware:
    """A plain ASGI middleware, not BaseHTTPMiddleware: the streamed replies and the client-disconnect handling stay as they are."""

    def __init__(self, app, key: str | None = None):
        self.app, self.key = app, check_key(key)
        self.refused = self.reported = 0   # refusals so far, and how many of them a log line has counted
        self.last_log = -LOG_EVERY
        log.info("access: %s", "remote requests need ORB_SHARE_KEY" if self.key else "private: remote requests are refused")

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        request = HTTPConnection(scope)  # the base of Request and WebSocket: both have headers, cookies and a URL
        if not is_remote(request):
            return await self.app(scope, receive, send)
        if self.key and same(given_key(request), self.key):
            return await self.app(scope, receive, send)
        if self.key and scope["type"] == "http" and scope["method"] == "GET" and same(request.query_params.get("key", ""), self.key):
            rest = [(k, v) for k, v in request.query_params.multi_items() if k != "key"]
            # one slash at the start: a path that starts with // would be a link to another site
            where = "/" + request.url.path.lstrip("/") + ("?" + urlencode(rest) if rest else "")
            response = RedirectResponse(where, status_code=303)  # no host in it: it is the one that the tunnel shows
            response.set_cookie(COOKIE, self.key, max_age=COOKIE_DAYS * 86400, httponly=True, samesite="lax", secure=is_https(request))
        else:
            self.refused += 1
            if time.monotonic() - self.last_log >= LOG_EVERY:  # the path only: a query can hold a wrong key
                log.warning("refused %d remote request(s), the last: %s %s (host %s)", self.refused - self.reported,
                            scope.get("method", "WS"), request.url.path, host_of(request))
                self.last_log, self.reported = time.monotonic(), self.refused
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            response = denied(request)
        await response(scope, receive, send)
