from __future__ import annotations

import pytest

from papaya.pidfile import PidFile, PidFileError


def test_pidfile_context_creates_and_removes(tmp_path):
    pid_path = tmp_path / "papaya.pid"

    with PidFile(pid_path) as handle:
        assert pid_path.exists()
        assert handle.pid == int(pid_path.read_text().strip())

    assert not pid_path.exists()


def test_pidfile_detects_running_process(monkeypatch, tmp_path):
    pid_path = tmp_path / "papaya.pid"
    pid_path.write_text("123", encoding="utf-8")

    monkeypatch.setattr("papaya.pidfile.pid_alive", lambda pid: True)

    with pytest.raises(PidFileError):
        PidFile.ensure_can_start(pid_path)


def test_pidfile_removes_stale_entries(monkeypatch, tmp_path):
    pid_path = tmp_path / "papaya.pid"
    pid_path.write_text("999", encoding="utf-8")

    calls: list[int] = []

    def _fake_pid_alive(pid: int) -> bool:
        calls.append(pid)
        return False

    monkeypatch.setattr("papaya.pidfile.pid_alive", _fake_pid_alive)

    PidFile.ensure_can_start(pid_path)

    assert not pid_path.exists()
    assert calls == [999]
