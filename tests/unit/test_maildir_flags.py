from __future__ import annotations

from papaya.maildir import (
    add_keyword_flag,
    build_maildir_filename,
    has_keyword_flag,
    parse_maildir_info,
    remove_keyword_flag,
)


def test_parse_maildir_info_breaks_out_components():
    assert parse_maildir_info("123.host:2,RSab") == ("123.host", "RS", "ab")
    assert parse_maildir_info("123.host:2,") == ("123.host", "", "")
    assert parse_maildir_info("123.host") == ("123.host", "", "")


def test_build_maildir_filename_sorts_flags():
    result = build_maildir_filename("123.host", "SR", "ba")
    assert result == "123.host:2,RSab"


def test_add_keyword_flag_adds_missing_letter():
    assert add_keyword_flag("123.host:2,RS", "a") == "123.host:2,RSa"


def test_add_keyword_flag_is_idempotent():
    assert add_keyword_flag("123.host:2,RSa", "a") == "123.host:2,RSa"


def test_add_keyword_flag_creates_flag_section_when_missing():
    assert add_keyword_flag("123.host", "a") == "123.host:2,a"


def test_remove_keyword_flag_strips_letter():
    assert remove_keyword_flag("123.host:2,RSab", "a") == "123.host:2,RSb"


def test_remove_keyword_flag_is_noop_when_missing():
    assert remove_keyword_flag("123.host:2,RS", "a") == "123.host:2,RS"


def test_has_keyword_flag_detects_presence():
    assert has_keyword_flag("123.host:2,RSab", "a") is True
    assert has_keyword_flag("123.host:2,RSab", "z") is False
