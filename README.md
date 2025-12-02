Papaya Spam Eater
==================

Papaya is an on-device daemon that keeps local Maildir inboxes clean. It watches configured accounts, extracts lightweight features from each message, and routes spam, newsletters, and ham based on pluggable machine learning classifiers plus dynamic sender allow/deny lists.

## Highlights

- 🔌 **Pluggable classifiers** – Naive Bayes (active) and TF‑IDF + SGD (shadow) with shared registry and persisted state.
- 📨 **Maildir-native pipeline** – Filesystem watcher reacts instantly to new mail or user-sorted messages and retrains in place.
- 📚 **Sender intelligence** – Ham/spam folder flags maintain per-account whitelists and blacklists to bypass ML when possible.
- 🧠 **Continuous learning** – Every user correction triggers immediate training; classifiers are saved atomically per account.
- 🛠 **Modern CLI** – Typer-based commands for daemon control, manual classification, bulk training, and classifier comparisons.
- 🔁 **Runtime control** – Foreground/background modes with PID management, SIGHUP config reloads, and SIGUSR1 status dumps.

## Requirements

- Python 3.10+
- `uv` (recommended) or `pip` for dependency management
- Maildir folders accessible on the local filesystem

Install dependencies in editable mode:

```bash
uv sync --all-extras
# or
pip install -e .
```

## Configuration

Papaya reads `~/.config/papaya/config.yaml` by default (override with `papaya --config path/to/config.yaml`). Minimal example:

```yaml
root_dir: ~/.local/lib/papaya
maildirs:
  - name: personal
    path: /var/vmail/example.com/personal
categories:
  Spam:
    min_confidence: 0.85
    flag: spam
  Newsletters:
    min_confidence: 0.7
  Important:
    min_confidence: 0.8
    flag: ham
classifiers:
  - name: naive_bayes
    type: naive_bayes
    mode: active
  - name: tfidf_sgd
    type: tfidf_sgd
    mode: shadow
logging:
  level: info
  debug_file: false
```

- `root_dir` stores models, sender lists, logs, and PID files.
- Categorised folders inherit behaviour from `flag` (`ham`, `spam`, or `neutral`).
- Add multiple Maildir accounts and classifier definitions as needed.

## Running the Daemon

Foreground mode (writes PID file and stays attached):

```bash
papaya daemon
```

Background mode spawns a child process and returns immediately:

```bash
papaya daemon --background
```

Useful flags:

- `--pid-file` – override PID file location (defaults to `<root_dir>/papaya.pid`).
- `--skip-initial-training` – start watching immediately without replaying existing category folders.

### Signals & Status

- `SIGTERM` / `SIGINT` – graceful shutdown.
- `SIGHUP` – reload configuration, rebuild watchers, and rerun initial training for any new accounts.
- `SIGUSR1` – log a status snapshot (watcher state, processed counts, routing breakdown).

Quick visibility into state is also available via:

```bash
papaya status
```

The status command inspects configured accounts, classifier persistence, PID liveness, and training metadata.

## CLI Reference

| Command | Description |
| --- | --- |
| `papaya daemon [-d]` | Run the daemon in foreground or background. |
| `papaya status` | Show configuration summary and whether the daemon is running. |
| `papaya classify path/to/message.eml [-a account]` | Inspect predictions from all classifiers for a single message. |
| `papaya train [--full] [-a account]` | Replay categorised mail for manual training; `--full` clears trained-ID cache first. |
| `papaya compare` | Analyse prediction logs vs. training truth to compare classifier accuracy. |

## Development

Papaya ships with Ruff, mypy, and pytest. Use the Just recipes to keep everything consistent:

```bash
just fix   # format + Ruff autofix
just check # format check, lint, mypy, pytest
```

Tests live under `tests/unit`. Integration/e2e suites will plug in once maildir fixtures are available. The documentation in `docs/prd.md` and `docs/plan.md` covers the full roadmap through Phase 10; optimisation, deep health monitoring, and hardening are intentionally deferred for this milestone.

---

Need help or spot a bug? File an issue with your config snippet and relevant log excerpts from `<root_dir>/logs/`.
