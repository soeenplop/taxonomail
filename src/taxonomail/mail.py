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


def _body(message: EmailMessage) -> str:
    part = message.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    try:
        content = part.get_content()
    except (LookupError, ValueError):
        payload = part.get_payload(decode=True)
        content = payload.decode("utf-8", "replace") if isinstance(payload, bytes) else ""
    if not isinstance(content, str):
        return ""
    if part.get_content_subtype() == "html":
        content = html_to_text(content)
    return _squash(content)


def parse_mail(raw: bytes) -> Mail:
    """Parse one raw message. Never raises on malformed mail; missing fields stay empty."""
    message = email.message_from_bytes(raw, policy=email.policy.default)
    assert isinstance(message, EmailMessage)
    sender = parseaddr(str(message.get("From", "")))[1].lower()
    message_id = str(message.get("Message-ID", "")).strip()
    return Mail(
        id=message_id or hashlib.sha256(raw).hexdigest()[:16],
        sender=sender,
        sender_domain=sender.rpartition("@")[2],
        subject=_squash(str(message.get("Subject", ""))),
        body=_body(message)[:BODY_CHARS],
    )
