from __future__ import annotations

import textwrap

from papaya.extractor import extract_features


def test_extract_features_prefers_plaintext_body() -> None:
    raw_email = textwrap.dedent(
        """
        From: Sender Name <sender@example.com>
        To: recipient@example.net
        Subject: =?utf-8?b?VGVzdCBFbWFpbA==?=
        List-Unsubscribe: <mailto:unsubscribe@example.com>
        X-Mailer: TestMailer
        MIME-Version: 1.0
        Content-Type: multipart/alternative; boundary="123"

        --123
        Content-Type: text/plain; charset="utf-8"

        Hello plain body.

        --123
        Content-Type: text/html; charset="utf-8"

        <html>
          <body>
            <p>Hello <strong>HTML</strong> body.</p>
            <a href="https://example.com">Example</a>
            <a href="https://evil.com">Evil</a>
          </body>
        </html>
        --123--
        """
    ).strip()

    features = extract_features(raw_email)

    assert features.body_text == "Hello plain body."
    assert features.subject == "Test Email"
    assert features.from_address == "sender@example.com"
    assert features.has_list_unsubscribe is True
    assert features.x_mailer == "TestMailer"
    assert features.link_count == 2
    assert features.domain_mismatch_score == 1
    assert features.is_malformed is False


def test_extract_features_falls_back_to_html_text() -> None:
    raw_email = textwrap.dedent(
        """
        From: Alerts <alerts@example.com>
        To: you@example.net
        Subject: Alerts
        MIME-Version: 1.0
        Content-Type: text/html; charset="utf-8"

        <html>
          <body>
            <p>Your account summary</p>
            <img src="cid:logo" />
            <form action="/submit"></form>
          </body>
        </html>
        """
    ).strip()

    features = extract_features(raw_email)

    assert "Your account summary" in features.body_text
    assert features.image_count == 1
    assert features.has_form is True
    assert features.link_count == 0


def test_extract_features_marks_malformed_payload_as_spammy() -> None:
    features = extract_features(b"\x00\xff\x00\xff")
    assert features.is_malformed is True
    assert features.body_text == ""
