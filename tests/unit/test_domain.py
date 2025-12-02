from papaya.extractor.domain import (
    domain_from_address,
    domain_mismatch_count,
    domains_from_links,
)


def test_domain_mismatch_counts_different_base_domains() -> None:
    links = [
        "https://example.com/welcome",
        "https://malicious.biz/login",
        "https://sub.example.com/news",
    ]

    count = domain_mismatch_count("Sender <sender@example.com>", links)

    assert count == 1  # only malicious.biz differs from example.com


def test_missing_sender_domain_marks_all_links_as_mismatch() -> None:
    count = domain_mismatch_count("No email header", ["https://example.com"])
    assert count == 1


def test_domains_from_links_handles_scheme_less_urls() -> None:
    domains = domains_from_links(["//static.example.net/assets", "example.com/page"])
    assert domains == ["example.net", "example.com"]


def test_domain_from_address_supports_ip_literals() -> None:
    domain = domain_from_address('"Device" <alerts@[2001:db8::1]>')
    assert domain == "2001:db8::1"
