"""Synthetic mail for tests."""

from __future__ import annotations

from email.message import EmailMessage


def raw_mail(
    subject: str, sender: str = "shop@example.org", body: str = "Hello.", n: int = 0
) -> bytes:
    """A synthetic RFC 822 message."""
    message = EmailMessage()
    message["From"] = sender
    message["To"] = "me@example.net"
    message["Subject"] = subject
    message["Message-ID"] = f"<m{n}@example.org>"
    message.set_content(body)
    return bytes(message)
