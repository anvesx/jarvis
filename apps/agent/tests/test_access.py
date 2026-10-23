"""The access gate: this computer is served, and a request from a tunnel or another device needs the key."""
import asyncio
import logging

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orb_agent.access import AccessMiddleware

KEY = "0123456789abcdef-0123456789"
TUNNEL = "https://quiet-river-1234.trycloudflare.com"
ran: list[str] = []


def make(key=None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AccessMiddleware, key=key)

    @app.get("/")
    def page():
        return PlainTextResponse("the page")

    @app.get("/v1/sessions")
    def sessions():
        return [{"id": "s1", "topic": "private talk"}]

    @app.post("/v1/sessions")
    def create():
        ran.append("create")
        return {"made": True}

    @app.get("/stream")
    def stream():
        return StreamingResponse((f"{i}\n" for i in range(3)), media_type="application/x-ndjson")

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        await socket.send_text("hello")
        await socket.close()

    return app


def remote(app, **kw):
    return TestClient(app, base_url=TUNNEL, follow_redirects=False, **kw)


def here(app, host="127.0.0.1:8100", **kw):
    return TestClient(app, base_url=f"http://{host}", **kw)


@pytest.fixture(autouse=True)
def reset():
    ran.clear()


@pytest.mark.parametrize("host", ["127.0.0.1:8100", "localhost:8100", "[::1]:8100", "localhost"])
@pytest.mark.parametrize("key", [None, KEY])
def test_this_computer_needs_no_key(host, key):
    c = here(make(key), host)
    assert c.get("/").text == "the page" and c.get("/v1/sessions").status_code == 200
    assert c.post("/v1/sessions").status_code == 200 and ran == ["create"]


def test_a_remote_request_is_refused_when_no_key_is_set():
    c = remote(make())
    page = c.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert page.status_code == 403 and page.headers["content-type"].startswith("text/html") and "private" in page.text
    api = c.get("/v1/sessions")
    assert api.status_code == 403 and "private" in api.json()["error"] and "private talk" not in api.text
    assert c.post("/v1/sessions").status_code == 403 and ran == []     # the handler did not run
    assert c.get("/?key=anything").status_code == 403
    assert c.get("/", headers={"x-orb-key": ""}).status_code == 403


def test_a_proxy_header_makes_a_loopback_host_remote():
    # a tunnel can keep the Host as localhost; the headers that it adds still tell
    c = here(make(KEY))
    for header in ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-forwarded-host", "x-real-ip", "forwarded"):
        assert c.get("/v1/sessions", headers={header: "203.0.113.9"}).status_code == 403, header
    assert c.get("/v1/sessions", headers={"x-forwarded-for": "203.0.113.9", "x-orb-key": KEY}).status_code == 200


def test_another_host_name_is_remote_also_when_it_points_here():
    # DNS rebinding: a web page makes its own name point at 127.0.0.1, and the browser sends that name as Host
    for host in ("evil.example:8100", "0.0.0.0:8100", "192.168.1.20:8100", "127.0.0.1.evil.example:8100"):
        assert here(make(KEY), host).get("/v1/sessions").status_code == 403, host


def test_the_key_lets_a_remote_request_in():
    c = remote(make(KEY))
    assert c.get("/v1/sessions").status_code == 403
    assert c.get("/v1/sessions", headers={"x-orb-key": KEY}).json()[0]["id"] == "s1"
    assert c.get("/v1/sessions", headers={"authorization": f"Bearer {KEY}"}).status_code == 200
    assert c.post("/v1/sessions", headers={"x-orb-key": KEY}).status_code == 200
    for wrong in (b"", b"x", KEY[:-1].encode(), KEY.encode() + b"x", KEY.upper().encode(), "é".encode("latin-1") * 30):
        assert c.get("/v1/sessions", headers={"x-orb-key": wrong}).status_code == 403, wrong   # bytes: a client may send any byte
    assert c.get("/v1/sessions", headers={"authorization": "Basic " + KEY}).status_code == 403


def test_the_link_with_the_key_sets_a_cookie_and_the_address_loses_the_key():
    c = remote(make(KEY))
    r = c.get(f"/?key={KEY}&tab=stt")
    assert r.status_code == 303 and r.headers["location"] == "/?tab=stt" and KEY not in r.headers["location"]
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("orb_key=") and "HttpOnly" in cookie and "SameSite=lax" in cookie and "Max-Age=2592000" in cookie
    assert "Secure" in cookie                           # the page came over https (a tunnel)
    assert c.get("/").text == "the page"                # the client keeps the cookie, so the next request has the key
    assert c.get("/v1/sessions").status_code == 200

    # behind a proxy: the service sees http, and the proxy says that the person used https
    def link(**headers):  # a new client each time: the cookie of one call must not open the next
        c = TestClient(make(KEY), base_url="http://quiet-river-1234.example.net", follow_redirects=False)
        return c.get(f"/?key={KEY}", headers=headers)

    assert "Secure" in link(**{"x-forwarded-proto": "https"}).headers["set-cookie"]
    assert "Secure" in link(**{"cf-visitor": '{"scheme":"https"}'}).headers["set-cookie"]
    assert "Secure" not in link().headers["set-cookie"]   # plain http: the browser would drop a Secure cookie
    assert link().headers["location"] == "/"


def test_the_link_works_only_with_the_right_key_and_only_for_a_page_load():
    c = remote(make(KEY))
    wrong = c.get("/?key=" + KEY[:-1])
    assert wrong.status_code == 403 and "set-cookie" not in wrong.headers
    post = c.post(f"/v1/sessions?key={KEY}")
    assert post.status_code == 403 and ran == []        # a link can open a page; it cannot start work
    assert c.get("/v1/sessions").status_code == 403     # no cookie came from the wrong tries


def test_the_redirect_never_leaves_the_site():
    # a path that starts with // would be read as a link to another site
    app, sent = make(KEY), []

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "//evil.example/x", "raw_path": b"//evil.example/x", "root_path": "",
             "query_string": f"key={KEY}".encode(), "scheme": "https", "http_version": "1.1", "server": ("h", 443),
             "client": ("203.0.113.9", 1), "headers": [(b"host", b"quiet-river-1234.trycloudflare.com")]}
    asyncio.run(app(scope, receive, send))
    start = sent[0]
    assert start["status"] == 303
    assert dict(start["headers"])[b"location"] == b"/evil.example/x"


def test_a_short_key_is_refused_and_a_blank_key_means_no_sharing():
    with pytest.raises(ValueError, match="16 or more"):
        AccessMiddleware(None, key="short")
    assert AccessMiddleware(None, key="   ").key is None and AccessMiddleware(None, key=None).key is None
    assert AccessMiddleware(None, key=f" {KEY} ").key == KEY


def test_a_refusal_is_logged_without_the_key_or_the_query(caplog):
    caplog.set_level(logging.INFO, logger="orb.agent")
    remote(make(KEY)).get("/v1/sessions?key=wrong-key-value-123&q=secret-words")
    text = caplog.text
    assert "refused 1 remote request(s), the last: GET /v1/sessions (host quiet-river-1234.trycloudflare.com)" in text
    for leak in ("wrong-key-value-123", "secret-words", KEY):
        assert leak not in text


def test_the_log_gets_one_line_for_each_period_with_the_count(caplog, monkeypatch):
    from orb_agent import access

    clock = [1000.0]
    monkeypatch.setattr(access.time, "monotonic", lambda: clock[0])
    caplog.set_level(logging.INFO, logger="orb.agent")
    c = remote(make())
    refusals = lambda: [r.getMessage() for r in caplog.records if r.getMessage().startswith("refused")]
    for _ in range(50):                     # a scanner: 50 requests in the same second
        c.get("/v1/sessions")
    assert len(refusals()) == 1 and refusals()[0].startswith("refused 1 remote")
    clock[0] += access.LOG_EVERY + 1        # the next period: one more line, with the 49 that it did not show
    c.get("/v1/sessions")
    assert len(refusals()) == 2 and refusals()[1].startswith("refused 50 remote")


def test_a_stream_passes_through_the_gate():
    r = remote(make(KEY)).get("/stream", headers={"x-orb-key": KEY})
    assert r.status_code == 200 and r.text.split() == ["0", "1", "2"]


def test_a_websocket_from_a_remote_client_is_closed_unless_it_has_the_key():
    # the test client always connects to ws://testserver, so each call names its own Host
    c = TestClient(make(KEY))
    tunnel, this_computer = {"host": "quiet-river-1234.trycloudflare.com"}, {"host": "127.0.0.1:8100"}
    with pytest.raises(WebSocketDisconnect) as refused:
        with c.websocket_connect("/ws", headers=tunnel):
            pass
    assert refused.value.code == 1008
    with c.websocket_connect("/ws", headers={**tunnel, "x-orb-key": KEY}) as ws:
        assert ws.receive_text() == "hello"
    with c.websocket_connect("/ws", headers=this_computer) as ws:
        assert ws.receive_text() == "hello"
