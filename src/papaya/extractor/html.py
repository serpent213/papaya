"""Helpers for analysing HTML email parts."""

from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class HtmlAnalysis:
    """Summary of structural HTML features."""

    text_content: str
    link_urls: list[str]
    link_count: int
    image_count: int
    has_form: bool


def analyse_html(html: str) -> HtmlAnalysis:
    """Return text content and structural metadata for an HTML fragment."""

    soup = BeautifulSoup(html, "lxml")

    text_content = soup.get_text(" ", strip=True)
    link_urls = _extract_links(soup)
    image_count = len(soup.find_all("img"))
    has_form = soup.find("form") is not None

    return HtmlAnalysis(
        text_content=text_content,
        link_urls=link_urls,
        link_count=len(link_urls),
        image_count=image_count,
        has_form=has_form,
    )


def _extract_links(soup: BeautifulSoup) -> list[str]:
    urls: list[str] = []
    for tag in soup.find_all(["a", "area"]):
        href = tag.get("href")
        if isinstance(href, str):
            normalized = href.strip()
            if normalized:
                urls.append(normalized)
    return urls


__all__ = ["HtmlAnalysis", "analyse_html"]
