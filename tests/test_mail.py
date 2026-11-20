from email.message import EmailMessage

import pytest

from taxonomail.mail import parse_mail
from tests.fakes import raw_mail


def test_plain_mail() -> None:
    mail = parse_mail(raw_mail("Your  invoice", "Shop <Billing@Example.org>", "Total:\n 12 EUR"))
    assert mail.id == "<m0@example.org>"
    assert mail.sender_domain == "example.org"
    assert mail.subject == "Your invoice"
    assert mail.body == "Total: 12 EUR"


def test_html_body_is_reduced_to_text() -> None:
    message = EmailMessage()
    message.set_content("<style>x{}</style><p>Hello <b>world</b></p>", subtype="html")
    assert parse_mail(bytes(message)).body == "Hello world"


def test_attachment_text_is_not_the_body() -> None:
    raw = (
        b"Subject: Hi\nContent-Type: multipart/mixed; boundary=b\n\n"
        b"--b\nContent-Type: text/plain; name=file.txt\n\nSECRET FILE\n"
        b"--b\nContent-Type: text/plain; charset=ascii\n\ncaf\xc3\xa9 body\n--b--\n"
    )
    assert parse_mail(raw).body == "café body"


@pytest.mark.parametrize(
    ("raw", "body"),
    [
        (b"From: " + b"(" * 1200 + b"\nSubject: Hi\n\nBody", "Body"),
        (b"Subject: Hi\nContent-Type: text/plain; x=" + b"(" * 1200 + b"\n\nBody", ""),
    ],
)
def test_pathological_headers_keep_subject(raw: bytes, body: str) -> None:
    mail = parse_mail(raw)
    assert (mail.sender, mail.subject, mail.body) == ("", "Hi", body)


def test_lf_and_crlf_give_the_same_mail() -> None:
    lf = (
        b"From: =?utf-8?q?J=C3=BCrgen?=\n =?utf-8?q?_M=C3=BCller?= <j@example.org>\n"
        b"Subject: Hi\nContent-Type: text/plain; charset=utf-8\n"
        b"Content-Transfer-Encoding: quoted-printable\n\n"
        b"line one=0D\nline two=0D\n"
    )
    crlf = lf.replace(b"\n", b"\r\n")
    assert parse_mail(lf) == parse_mail(crlf)
    assert parse_mail(lf).sender == "j@example.org"


def test_forwarded_mail_and_nul_charset() -> None:
    forwarded = (
        b"Subject: Fwd\nContent-Type: multipart/mixed; boundary=b\n\n"
        b"--b\nContent-Type: text/html\n\n<p>Outer</p>\n"
        b"--b\nContent-Type: message/rfc822\n\nSubject: In\n\nInner text\n--b--\n"
    )
    assert parse_mail(forwarded).body == "Outer"
    nul = b'Subject: Hi\nContent-Type: text/plain; charset="utf\x00-8"\n\ncaf\xc3\xa9'
    assert parse_mail(nul).body == "café"
