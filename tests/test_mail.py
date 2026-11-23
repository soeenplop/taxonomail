from datetime import UTC, datetime
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


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("Mon, 01 Jan 2024 10:00:00 +0100", datetime(2024, 1, 1, 9, tzinfo=UTC)),
        ("Mon, 01 Jan 2024 10:00:00 -0000", datetime(2024, 1, 1, 10, tzinfo=UTC)),
        ("Mon, 01 Jan 2024 10:00:00 -0000", datetime(2024, 1, 1, 10, tzinfo=UTC)),
        ("Mon, 01 Jan 2024 10:00:00 +0000 (UTC)", datetime(2024, 1, 1, 10, tzinfo=UTC)),
        ("Mon, 01 Jan 2024 10:00:00", None),
        ("Mon, 01 Jan 2024 10:00:00 CET", None),
        ("Mon, 01 Jan 2024 10:00:00 XYZ", None),
        ("Mon, 1 Jan 2024 10:00:00 XYZ (-0000)", None),
        ("Mon, 1 Jan 2024 10:00:00 XYZ (local time 09:00 -0000 at origin)", None),
        ("Mon, 1 Jan 2024 10:00:00 (a (b) 10:00 -0000) XYZ", None),
        (
            "Mon, 1 Jan 2024 10:00:00 -0000 (a (nested) comment)",
            datetime(2024, 1, 1, 10, tzinfo=UTC),
        ),
        ("Mon, 1 Jan 2024 10:00 -0000 (unknown)", datetime(2024, 1, 1, 10, tzinfo=UTC)),
        ("31 Dec 9999 23:59:59 -0100", None),
        ("not a date", None),
        (None, None),
    ],
)
def test_date_is_read_in_utc(date: str | None, expected: datetime | None) -> None:
    # Raw bytes: building the mail with EmailMessage would already normalize the zone.
    head = b"" if date is None else b"Date: " + date.encode() + b"\n"
    raw = head + b"From: a@example.org\nSubject: s\n\nbody"
    assert parse_mail(raw).date == expected


def test_two_dates_mean_no_date() -> None:
    raw = (
        b"Date: Mon, 01 Jan 2024 10:00:00 +0000\nDate: Mon, 01 Jan 2099 10:00:00 +0000\n"
        b"From: a@example.org\nSubject: s\n\nbody"
    )
    assert parse_mail(raw).date is None


def test_deeply_nested_date_comment_is_fast() -> None:
    date = "Mon, 1 Jan 2024 10:00:00 XYZ " + "(" * 50_000 + "-0000" + ")" * 50_000
    raw = b"Date: " + date.encode() + b"\nFrom: a@example.org\nSubject: s\n\nbody"
    assert parse_mail(raw).date is None


def test_escaped_parenthesis_stays_inside_the_comment() -> None:
    raw = b"Date: Mon, 1 Jan 2024 10:00:00 XYZ (a \\) 10:00 -0000)\nSubject: s\n\nbody"
    assert parse_mail(raw).date is None
