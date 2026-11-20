"""Turn raw RFC 822 messages into the few fields the pipeline uses."""

from __future__ import annotations

import email
import email.policy
import hashlib
import re
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import parseaddr
from html.parser import HTMLParser

BODY_CHARS = 1500
"""Characters of body text kept per mail."""


@dataclass(frozen=True)
class Mail:
    """One mail, reduced to what clustering and naming need."""

    id: str
    """Message-ID, or a hash of the raw bytes when there is none."""
    sender: str
    sender_domain: str
    subject: str
    body: str

    def text(self) -> str:
        """The text that is embedded and shown to the LLM."""
        return f"From: {self.sender_domain}\nSubject: {self.subject}\n\n{self.body}"


class _TextExtractor(HTMLParser):
    """Collect visible text from HTML, skipping scripts and styles."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Visible text of an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    return " ".join(parser.parts)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_attachment(part: EmailMessage) -> bool:
    try:
        return bool(part.get_filename()) or part.get_content_disposition() == "attachment"
    except Exception:  # malformed headers: treat as not an attachment
        return False


def _candidates(part: EmailMessage, found: dict[str, EmailMessage]) -> None:
    """Collect the first plain and html body parts, skipping attachments and their children."""
    if _is_attachment(part):
        return
    if part.is_multipart():
        for child in part.iter_parts():
            assert isinstance(child, EmailMessage)
            _candidates(child, found)
    elif part.get_content_type() in {"text/plain", "text/html"}:
        found.setdefault(part.get_content_subtype(), part)


def _decode(part: EmailMessage) -> str:
    payload = part.get_payload(decode=True)
    if not isinstance(payload, bytes):
        return ""
    charset = part.get_content_charset() or "utf-8"
    for encoding in (charset, "utf-8", "cp1252"):
        try:
            return payload.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return payload.decode("utf-8", "replace")


def _body(message: EmailMessage) -> str:
    found: dict[str, EmailMessage] = {}
    _candidates(message, found)
    part = found.get("plain") or found.get("html")
    if part is None:
        return ""
    content = _decode(part)
    if part.get_content_subtype() == "html":
        content = html_to_text(content)
    return _squash(content)


def _header(message: EmailMessage, name: str) -> str:
    """A header as text; empty when the header parser fails (e.g. pathological nesting)."""
    try:
        return str(message.get(name, ""))
    except (RecursionError, ValueError, TypeError, IndexError):
        return ""


def _address(value: str) -> str:
    try:
        return parseaddr(value)[1].lower()
    except (RecursionError, ValueError):
        return ""


def parse_mail(raw: bytes) -> Mail:
    """Parse one raw message. Never raises on malformed mail; missing fields stay empty."""
    message = email.message_from_bytes(raw, policy=email.policy.default)
    assert isinstance(message, EmailMessage)
    sender = _address(_header(message, "From"))
    message_id = _header(message, "Message-ID").strip()
    return Mail(
        id=message_id or hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()[:16],
        sender=sender,
        sender_domain=sender.rpartition("@")[2],
        subject=_squash(_header(message, "Subject")),
        body=_body(message)[:BODY_CHARS],
    )
