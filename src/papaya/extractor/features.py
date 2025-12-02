"""Feature extraction from RFC822 email messages."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from email import policy
from email.header import decode_header, make_header
from email.message import Message
from email.parser import BytesParser
from email.utils import parseaddr

from ..types import Features
from .domain import domain_mismatch_count
from .html import analyse_html


def extract_features(raw_message: bytes | str | Message) -> Features:
    """Parse raw email data into a structured Features instance."""

    message, is_malformed = _parse_message(raw_message)
    if message is None:
        return _malformed_features()

    subject = _decode_header_value(message.get("Subject", ""))
    from_display, from_address = parseaddr(message.get("From", ""))
    if not from_address:
        is_malformed = True

    text_bodies: list[str] = []
    html_texts: list[str] = []
    link_targets: list[str] = []
    link_count = 0
    image_count = 0
    has_form = False

    for part in _iter_body_parts(message):
        payload = part.get_payload(decode=True)
        if payload is None or not isinstance(payload, (bytes, bytearray)):
            continue
        payload_bytes = bytes(payload)
        decoded = _decode_bytes(payload_bytes, part.get_content_charset())
        maintype = part.get_content_maintype()
        subtype = part.get_content_subtype()
        content_type = f"{maintype}/{subtype}".lower()

        if content_type == "text/plain":
            text_bodies.append(decoded)
        elif content_type == "text/html":
            html_texts.append(decoded)
            html_info = analyse_html(decoded)
            link_targets.extend(html_info.link_urls)
            link_count += html_info.link_count
            image_count += html_info.image_count
            has_form = has_form or html_info.has_form

    body_text = _select_body_text(text_bodies, html_texts)
    has_list_unsubscribe = bool(message.get("List-Unsubscribe"))
    x_mailer = message.get("X-Mailer") or message.get("User-Agent")
    domain_mismatch = domain_mismatch_count(from_address, link_targets)

    return Features(
        body_text=body_text,
        subject=subject,
        from_address=from_address,
        from_display_name=from_display,
        has_list_unsubscribe=has_list_unsubscribe,
        x_mailer=x_mailer,
        link_count=link_count,
        image_count=image_count,
        has_form=has_form,
        domain_mismatch_score=float(domain_mismatch),
        is_malformed=is_malformed,
    )


def _parse_message(raw_message: bytes | str | Message) -> tuple[Message | None, bool]:
    if isinstance(raw_message, Message):
        if not tuple(raw_message.keys()):
            return None, True
        return raw_message, False

    try:
        if isinstance(raw_message, bytes):
            message = BytesParser(policy=policy.default).parsebytes(raw_message)
        else:
            message = BytesParser(policy=policy.default).parsebytes(
                raw_message.encode("utf-8", errors="ignore")
            )
    except Exception:
        return None, True
    if not tuple(message.keys()):
        return None, True
    return message, False


def _iter_body_parts(message: Message) -> Iterable[Message]:
    for part in message.walk():
        if part.is_multipart():
            continue
        content_disposition = (part.get_content_disposition() or "").lower()
        if content_disposition == "attachment":
            continue
        yield part


def _decode_bytes(data: bytes, charset: str | None) -> str:
    candidates: Sequence[str] = []
    if charset:
        candidates = [charset]
    candidates = list(candidates) + ["utf-8", "latin-1"]
    for encoding in candidates:
        try:
            return data.decode(encoding, errors="replace")
        except LookupError:
            continue
    return data.decode("utf-8", errors="ignore")


def _select_body_text(plain_bodies: list[str], html_texts: list[str]) -> str:
    source = plain_bodies if plain_bodies else html_texts
    if not source:
        return ""
    combined = "\n".join(line.strip() for line in source if line.strip())
    return combined.strip()


def _decode_header_value(value: str) -> str:
    try:
        header = make_header(decode_header(value))
        decoded = str(header)
    except Exception:
        decoded = value
    return decoded.strip()


def _malformed_features() -> Features:
    return Features(
        body_text="",
        subject="",
        from_address="",
        from_display_name="",
        has_list_unsubscribe=False,
        x_mailer=None,
        link_count=0,
        image_count=0,
        has_form=False,
        domain_mismatch_score=0.0,
        is_malformed=True,
    )


__all__ = ["extract_features"]
