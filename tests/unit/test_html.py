from papaya.extractor.html import analyse_html


def test_analyse_html_counts_links_images_and_forms() -> None:
    html = """
    <html>
      <body>
        <p>Hello <strong>world</strong></p>
        <a href="https://example.com">Example</a>
        <a href=" https://news.example.com ">Newsletter</a>
        <img src="cid:1" />
        <form action="/submit"></form>
      </body>
    </html>
    """

    analysis = analyse_html(html)

    assert "Hello world" in analysis.text_content
    assert analysis.link_count == 2
    assert analysis.image_count == 1
    assert analysis.has_form is True
    assert analysis.link_urls == ["https://example.com", "https://news.example.com"]
