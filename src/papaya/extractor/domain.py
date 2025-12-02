"""Domain extraction utilities used for phishing heuristics."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable
from email.utils import parseaddr
from urllib.parse import urlparse

HOST_RE = re.compile(r"^[A-Za-z0-9.-]+$")


def domain_from_address(address: str) -> str | None:
    """Return the normalized sender domain for a From address."""

    _, email_addr = parseaddr(address or "")
    if not email_addr and address:
        if "<" in address and ">" in address:
            email_addr = address.split("<", 1)[1].split(">", 1)[0].strip()
        else:
            email_addr = address.strip()
    if not email_addr or "@" not in email_addr:
        return None
    return _normalize_host(email_addr.split("@", 1)[1])


def domains_from_links(links: Iterable[str]) -> list[str]:
    """Extract normalized hostnames for each URL."""

    domains: list[str] = []
    for link in links:
        domain = _domain_from_link(link)
        if domain:
            domains.append(domain)
    return domains


def domain_mismatch_count(from_address: str, links: Iterable[str]) -> int:
    """Count links whose domains do not match the sender's domain."""

    sender_domain = domain_from_address(from_address or "")
    link_domains = domains_from_links(links)
    if not link_domains:
        return 0
    if sender_domain is None:
        return len(link_domains)
    mismatched = sum(1 for domain in link_domains if domain != sender_domain)
    return mismatched


def _domain_from_link(link: str) -> str | None:
    parsed = urlparse(link)
    host = parsed.hostname
    if not host and parsed.scheme == "" and parsed.path:
        # URLs without scheme like "//example.com" or "example.com"
        fallback = urlparse(f"http://{link}")
        host = fallback.hostname
    return _normalize_host(host)


def _normalize_host(host: str | None) -> str | None:
    if not host:
        return None
    candidate = host.strip().lower().rstrip(".")
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if not candidate:
        return None

    # IPv4/IPv6 literals retain their exact string.
    try:
        ipaddress.ip_address(candidate)
        return candidate
    except ValueError:
        pass

    if not HOST_RE.match(candidate):
        return None

    labels = [label for label in candidate.split(".") if label]
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0]
    return ".".join(labels[-2:])


__all__ = ["domain_mismatch_count", "domain_from_address", "domains_from_links"]
