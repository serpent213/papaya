from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from papaya.cli import app
from papaya.maildir import category_subdir, ensure_maildir_structure
from papaya.store import Store

runner = CliRunner()


def _write_config(tmp_path: Path, maildir: Path, root_dir: Path | None = None) -> Path:
    root = root_dir or (tmp_path / "state")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                f"root_dir: {root}",
                "rules: |",
                "  pass",
                "train: |",
                "  pass",
                "maildirs:",
                "  - name: personal",
                f"    path: {maildir}",
                "categories:",
                "  Spam:",
                "    flag: spam",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config


def _sample_message(path: Path, *, message_id: str = "<msg-1>") -> None:
    contents = (
        "From: sender@example.com\n"
        "To: recipient@example.com\n"
        "Subject: Test\n"
        f"Message-ID: {message_id}\n"
        "\n"
        "Hello world"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_status_reports_accounts(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    config_path = _write_config(tmp_path, maildir)

    result = runner.invoke(app, ["-c", str(config_path), "status"])

    assert result.exit_code == 0
    assert "personal" in result.stdout
    assert str(maildir) in result.stdout


def test_classify_outputs_decision(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    config_path = _write_config(tmp_path, maildir)
    message_path = tmp_path / "sample.eml"
    _sample_message(message_path, message_id="<sample-1>")

    result = runner.invoke(app, ["-c", str(config_path), "classify", str(message_path)])

    assert result.exit_code == 0
    assert "Decision:" in result.stdout
    assert "category:" in result.stdout


def test_train_full_replays_categorised_mail(tmp_path):
    maildir = tmp_path / "Maildir"
    ensure_maildir_structure(maildir, ["Spam"])
    config_path = _write_config(tmp_path, maildir)
    spam_message = category_subdir(maildir, "Spam", "new") / "spam1"
    _sample_message(spam_message, message_id="<spam-123>")

    result = runner.invoke(app, ["-c", str(config_path), "train", "--full"])

    assert result.exit_code == 0
    store = Store(tmp_path / "state")
    snapshot = store.trained_ids.snapshot()
    assert "<spam-123>" in snapshot
    assert snapshot["<spam-123>"] == "Spam"
