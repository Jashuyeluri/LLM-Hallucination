"""Multi-source ingestion for trusted, user-supplied knowledge sources.

Only public HTTPS URLs are fetched. Redirects are revalidated to reduce SSRF risk.
This is not a general web crawler or search engine connector.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

MAX_URL_BYTES = 2_000_000
TIMEOUT_SECONDS = 8


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header"}
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth:
            value = " ".join(data.split())
            if value:
                self.parts.append(value)


def _resolve_public(host: str) -> str:
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError(f"Could not resolve source host: {host}") from exc
    if not addresses:
        raise ValueError("Source host did not resolve.")
    chosen = None
    for item in addresses:
        ip = ipaddress.ip_address(item[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("Private, loopback, link-local and reserved network addresses are blocked.")
        chosen = chosen or str(ip)
    return chosen


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, pinned_ip, **kwargs):
        super().__init__(host, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        host = urllib.parse.urlsplit(req.full_url).hostname
        pinned_ip = _resolve_public(host.rstrip(".").lower())
        return self.do_open(lambda target, **kwargs: _PinnedHTTPSConnection(target, pinned_ip, **kwargs), req, context=self._context)


def _validate_public_https(url: str) -> str:
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("Only absolute HTTPS source URLs are allowed.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed.")
    if parsed.port not in (None, 443):
        raise ValueError("Only the standard HTTPS port is allowed.")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are not allowed.")
    _resolve_public(host)
    return urllib.parse.urlunsplit(("https", parsed.netloc, parsed.path or "/", parsed.query, ""))


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_url_text(url: str) -> dict:
    safe_url = _validate_public_https(url)
    opener = urllib.request.build_opener(_SafeRedirect, _PinnedHTTPSHandler)
    request = urllib.request.Request(
        safe_url,
        headers={"User-Agent": "HallucinationVerificationLab/1.0", "Accept": "text/html,text/plain"},
    )
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            content_type = response.headers.get_content_type()
            if content_type not in {"text/html", "text/plain"}:
                raise ValueError(f"Unsupported URL content type: {content_type}")
            body = response.read(MAX_URL_BYTES + 1)
            if len(body) > MAX_URL_BYTES:
                raise ValueError("URL source exceeds the 2 MB limit.")
            charset = response.headers.get_content_charset() or "utf-8"
            raw = body.decode(charset, errors="replace")
            if content_type == "text/html":
                parser = _TextExtractor()
                parser.feed(raw)
                text = "\n".join(parser.parts)
            else:
                text = raw
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if not text.strip():
                raise ValueError("No readable text was found at the URL.")
            return {"url": response.geturl(), "text": text[:60000], "content_type": content_type}
    except urllib.error.URLError as exc:
        raise ValueError(f"Could not fetch source URL: {exc.reason}") from exc


def combine_sources(sources: list[dict], max_chars: int = 60000) -> tuple[str, list[dict]]:
    """Combine source texts with explicit provenance markers and bounded total size."""
    sections, manifest, used = [], [], 0
    if len(sources) == 1:
        source = sources[0]
        name = str(source.get("name") or "Source 1").replace("\\n", " ").strip()[:180]
        content = str(source.get("text") or "").strip()
        if not content:
            return "", []
        return content[:max_chars], [{
            "id": 1, "name": name, "kind": source.get("kind", "document"),
            "url": source.get("url"), "start": 0,
            "end": min(len(content), max_chars),
            "characters": min(len(content), max_chars),
        }]
    for index, source in enumerate(sources, start=1):
        name = str(source.get("name") or f"Source {index}").replace("\n", " ").strip()[:180]
        content = str(source.get("text") or "").strip()
        if not content:
            continue
        header = f"\n\n[DOCUMENT {index}: {name}]\n"
        remaining = max_chars - used - len(header)
        if remaining <= 0:
            break
        content = content[:remaining]
        start = used + len(header)
        sections.append(header + content)
        manifest.append({
            "id": index,
            "name": name,
            "kind": source.get("kind", "document"),
            "url": source.get("url"),
            "start": start,
            "end": start + len(content),
            "characters": len(content),
        })
        used += len(header) + len(content)
    return "".join(sections), manifest
