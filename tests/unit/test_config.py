from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from papaya.config import ConfigError, load_config


def _write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return config_path


def test_load_config_success(tmp_path: Path) -> None:
    maildir = tmp_path / "maildir"
    maildir.mkdir()
    config_path = _write_config(
        tmp_path,
        f"""
        rootdir: {tmp_path}/state
        module_paths:
          - ~/custom/papaya
        rules: |
          skip()
        train_rules: |
          pass
        maildirs:
          - name: personal
            path: {maildir}
            rules: |
              fallback()
            train_rules: |
              pass
        categories:
          Spam: {{}}
        logging:
          level: debug
          debug_file: true
        """,
    )

    config = load_config(config_path)

    assert config.root_dir == (tmp_path / "state")
    assert config.maildirs[0].name == "personal"
    assert config.maildirs[0].rules is not None
    assert config.module_paths == [Path("~/custom/papaya").expanduser()]
    assert "skip()" in config.rules
    assert "pass" in config.train_rules
    assert config.categories["Spam"].name == "Spam"
    assert config.logging.debug_file is True


def test_load_config_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        rules: |
          pass
        train_rules: |
          pass
        maildirs:
          - name: inbox
            path: ~/Maildir
        categories:
          Spam: {}
        """,
    )

    monkeypatch.setenv("PAPAYA_CONFIG", str(config_path))
    config = load_config()
    assert config.maildirs[0].name == "inbox"


@pytest.mark.parametrize(
    "bad_content, expected_message",
    [
        (
            """
            rules: |
              pass
            train_rules: |
              pass
            maildirs: []
            categories:
              Spam: {}
            """,
            "At least one maildir must be configured",
        ),
        (
            """
            rules: |
              pass
            train_rules: |
              pass
            maildirs:
              - name: inbox
                path: ~/Maildir
            categories:
              Spam: []
            """,
            "Category 'Spam' config must be a mapping",
        ),
        (
            """
            train_rules: |
              pass
            maildirs:
              - name: inbox
                path: ~/Maildir
            categories:
              Spam: {}
            """,
            "rules must be provided",
        ),
        (
            """
            rules: |
            train_rules: |
              pass
            maildirs:
              - name: inbox
                path: ~/Maildir
            categories:
              Spam: {}
            """,
            "rules cannot be empty",
        ),
        (
            """
            rules: |
              pass
            train_rules: |
              pass
            maildirs:
              - name: inbox
                path: ~/Maildir
                rules: 123
            categories:
              Spam: {}
            """,
            "maildirs[1].rules must be a string",
        ),
        (
            """
            rules: |
              pass
            train_rules: |
              pass
            categories:
              Spam: {}
            """,
            "At least one maildir must be configured",
        ),
    ],
)
def test_load_config_validation_errors(
    bad_content: str,
    expected_message: str,
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path, bad_content)
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_path)
    assert expected_message in str(excinfo.value)
