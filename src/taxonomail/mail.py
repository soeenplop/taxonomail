"""Turn raw RFC 822 messages into the few fields the pipeline uses."""

from __future__ import annotations

import email
import email.parser
import email.policy
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import parseaddr, parsedate_to_datetime
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
    date: datetime | None = None
    """From the Date header, in UTC; ``None`` when missing or unreadable."""

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
    if _is_attachment(part) or part.get_content_type() == "message/rfc822":
        return
    if part.get_content_maintype() == "multipart":
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
        except (LookupError, ValueError):
            continue
    return payload.decode("utf-8", "replace")


def _body(message: EmailMessage) -> str:
    try:
        return _extract_body(message)
    except Exception:  # one bad mail must never abort a run
        return ""


def _extract_body(message: EmailMessage) -> str:
    found: dict[str, EmailMessage] = {}
    _candidates(message, found)
    part = found.get("plain") or found.get("html")
    if part is None:
        return ""
    content = _decode(part)
    if part.get_content_subtype() == "html":
        content = html_to_text(content)
    return _squash(content)


def _header(message: Message, name: str) -> str:
    """A header as text; empty when the header parser fails (e.g. pathological nesting)."""
    try:
        value = message.get(name, "")
        if isinstance(message.policy, email.policy.Compat32):
            value = str(make_header(decode_header(str(value))))
        return str(value)
    except Exception:
        return ""


def _address(value: str) -> str:
    try:
        return parseaddr(value)[1].lower()
    except (RecursionError, ValueError):
        return ""


def _date(message: Message) -> datetime | None:
    try:
        parsed = parsedate_to_datetime(_header(message, "Date"))
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fields(raw: bytes, message: Message, body: str) -> Mail:
    sender = _address(_header(message, "From"))
    message_id = _header(message, "Message-ID").strip()
    return Mail(
        id=message_id or hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()[:16],
        sender=sender,
        sender_domain=sender.rpartition("@")[2],
        subject=_squash(_header(message, "Subject")),
        body=body[:BODY_CHARS],
        date=_date(message),
    )


def parse_mail(raw: bytes) -> Mail:
    """Parse one raw message. Never raises on malformed mail; missing fields stay empty."""
    try:
        message = email.message_from_bytes(raw, policy=email.policy.default)
        assert isinstance(message, EmailMessage)
        return _fields(raw, message, _body(message))
    except Exception:  # one bad mail must never abort a run
        pass
    try:
        head = re.split(rb"\r?\n\r?\n", raw, maxsplit=1)[0]
        # compat32 parses headers lazily, so one broken header cannot fail the others.
        legacy = email.parser.BytesHeaderParser().parsebytes(head)
        return _fields(raw, legacy, "")
    except Exception:
        return Mail(
            id=hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()[:16],
            sender="",
            sender_domain="",
            subject="",
            body="",
        )
