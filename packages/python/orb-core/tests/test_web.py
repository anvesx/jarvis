import socket

import httpx

from orb_core import web


def fake_resolve(ip):
    def resolve(host, port, proto=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, proto, "", (ip, port))]
    return resolve


def test_find_url_strips_trailing_punctuation():
    assert web.find_url("Let's discuss https://example.com/a-post). Thanks") == "https://example.com/a-post"
    assert web.find_url("no link here") is None


def test_check_public_blocks_private_and_odd_schemes(monkeypatch):
    monkeypatch.delenv("ORB_ALLOW_PRIVATE_URLS", raising=False)
    assert web.check_public("ftp://example.com/x") is not None
    assert web.check_public("http:///nohost") is not None
    for ip in ["127.0.0.1", "10.1.2.3", "192.168.1.5", "169.254.169.254", "::1"]:
        assert "private" in web.check_public("http://evil.test/", resolve=fake_resolve(ip))
    assert web.check_public("https://example.com/", resolve=fake_resolve("93.184.215.14")) is None


def test_check_public_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("ORB_ALLOW_PRIVATE_URLS", "1")
    assert web.check_public("http://127.0.0.1:8100/") is None


def test_extract_keeps_text_and_skips_navigation():
    html = """<html><head><title>GST &amp; you</title></head><body>
    <nav><p>Menu menu menu menu menu menu menu menu menu</p></nav>
    <h1>Input tax credit</h1>
    <p>You can claim input tax credit on goods used for business, with a valid invoice.</p>
    <script>var x = "ignore me, I am a script";</script>
    <p>short</p></body></html>"""
    title, text = web.extract(html)
    assert title == "GST & you"
    assert "## Input tax credit" in text and "valid invoice" in text
    assert "Menu" not in text and "script" not in text and "short" not in text


def test_load_rechecks_every_redirect(monkeypatch):
    monkeypatch.delenv("ORB_ALLOW_PRIVATE_URLS", raising=False)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})

    checked = []

    def fake_check(url, resolve=None):
        checked.append(url)
        return None if "169.254" not in url else "the host resolves to a private address (169.254.169.254), so it is not loaded"

    monkeypatch.setattr(web, "check_public", fake_check)
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    out = web.load("https://public.example/post", client=client)
    assert "private address" in out["error"]
    assert calls == ["https://public.example/post"]  # the private hop is never requested
    assert checked == ["https://public.example/post", "http://169.254.169.254/latest/meta-data"]
