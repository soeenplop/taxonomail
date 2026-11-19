from email.message import EmailMessage

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
