"""Load the readable text of a web page, so an agent can discuss the real text and not guess it.

Keeps the page title and the text of headings, paragraphs, list items, quotes and code blocks. Skips scripts,
styles, navigation, headers, footers and forms. The text is reference material for the LLM, never instructions.

Security: a URL comes from user input, so the server must not fetch private addresses (127.0.0.1, 10.x, cloud
metadata at 169.254.169.254 and so on). Every hop of a redirect is checked again. Set ORB_ALLOW_PRIVATE_URLS=1
only on a machine where that is safe.
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

URL_RE = re.compile(r"https?://[^\s<>\"']+")
MAX_BYTES = 6_000_000
MAX_CHARS = 40_000  # about 10k tokens: enough for a long blog post, and it fits Gemma's context with the chat
MAX_REDIRECTS = 5
KEEP = {"h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "figcaption"}
SKIP = {"script", "style", "noscript", "nav", "header", "footer", "form", "svg", "aside", "button"}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.blocks, self.skip, self.open, self.buf, self.in_title = "", [], 0, [], [], False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag in KEEP and not self.skip:
            self.open.append(tag)
            self.buf.append("")
        elif tag == "meta" and not self.title:
            a = dict(attrs)
            if a.get("property") == "og:title" and a.get("content"):
                self.title = a["content"]

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self.in_title = False
        elif self.open and tag == self.open[-1]:
            self.open.pop()
            text = " ".join(self.buf.pop().split())
            if self.buf:  # a block inside a block: give the text to the outer one
                self.buf[-1] += " " + text
            elif text and (tag.startswith("h") or len(text) >= 30):
                self.blocks.append(("## " if tag in ("h1", "h2", "h3", "h4") else "") + text)

    def handle_data(self, data):
        if self.in_title and not self.title:
            self.title = data.strip()
        elif self.buf and not self.skip:
            self.buf[-1] += data


def find_url(text: str) -> str | None:
    m = URL_RE.search(text or "")
    return m.group(0).rstrip(").,;") if m else None


def check_public(url: str, resolve=socket.getaddrinfo) -> str | None:
    """None when the URL may be fetched, else the reason why not."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        return "only http and https links can be loaded"
    if not parts.hostname:
        return "the link has no host name"
    if os.environ.get("ORB_ALLOW_PRIVATE_URLS") == "1":
        return None
    try:
        infos = resolve(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError) as e:
        return f"the host name does not resolve ({type(e).__name__})"
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if not addr.is_global:
            return f"the host resolves to a private address ({addr}), so it is not loaded"
    return None


def extract(html: str) -> tuple[str, str]:
    """The title and the readable text of an HTML page."""
    p = _Text()
    p.feed(html)
    blocks, seen = [], set()
    for b in p.blocks:
        if b not in seen:
            seen.add(b)
            blocks.append(b)
    return unescape(p.title).strip(), "\n\n".join(blocks)


def load(url: str, client: httpx.Client | None = None) -> dict:
    """Returns {url, title, text, words, truncated} or {url, error}. It never raises for network errors."""
    own = client is None
    client = client or httpx.Client(follow_redirects=False, timeout=20, headers={"User-Agent": UA})
    try:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            reason = check_public(current)
            if reason:
                return {"url": url, "error": reason}
            r = client.get(current)
            if r.is_redirect and r.headers.get("location"):
                current = urljoin(current, r.headers["location"])
                continue
            break
        else:
            return {"url": url, "error": f"more than {MAX_REDIRECTS} redirects"}
        if r.status_code != 200:
            return {"url": url, "error": f"HTTP {r.status_code}"}
        if "html" not in r.headers.get("content-type", "html"):
            return {"url": url, "error": f"not an HTML page ({r.headers.get('content-type')})"}
        html = r.text[:MAX_BYTES]
    except httpx.HTTPError as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}
    finally:
        if own:
            client.close()
    title, text = extract(html)
    if len(text) < 200:
        return {"url": url, "error": "the page has almost no readable text (it may need JavaScript)"}
    cut = text[:MAX_CHARS]
    return {"url": url, "title": title, "text": cut, "words": len(cut.split()), "truncated": len(text) > MAX_CHARS}
