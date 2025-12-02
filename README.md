Papaya Spam Eater
==================

Papaya is an on-device daemon that keeps local Maildir inboxes clean. It watches configured accounts, extracts lightweight features from each message, and routes spam, newsletters, and ham via a rule engine that orchestrates pluggable ML modules plus dynamic sender allow/deny lists.

## Highlights

- 🧠 **Rule-driven automation** – Python snippets embedded in config files coordinate module calls, sender shortcuts, and routing decisions.
- 🔌 **Hot-reloadable modules** – Built-in and user modules expose classify/train hooks with persisted state managed through the store.
- 📚 **Sender intelligence** – Ham/spam folder flags maintain per-account whitelists and blacklists to bypass ML when possible.
- 🔁 **Continuous learning** – Every user correction triggers immediate training through `train_rules`; module state is persisted incrementally.
- 🛠 **Modern CLI** – Typer-based commands for daemon control, manual classification, and bulk training.
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
module_paths:
  - ~/.config/papaya/modules
maildirs:
  - name: personal
    path: /var/vmail/example.com/personal
rules: |
  features = modules.extract_features.classify(message)
  bayes = modules.naive_bayes.classify(message, features, account)

  if bayes.scores.get("Spam", 0) > 0.85:
      move_to("Spam", confidence=bayes.scores["Spam"])
  if features.has_list_unsubscribe:
      move_to("Newsletters", confidence=0.8)

train_rules: |
  features = modules.extract_features.classify(message)
  modules.naive_bayes.train(message, features, category, account)
  modules.tfidf_sgd.train(message, features, category, account)
categories:
  Spam:
    flag: spam
  Newsletters:
    flag: neutral
  Important:
    flag: ham
logging:
  level: info
  debug_file: false
```

- `root_dir` stores models, sender lists, logs, and PID files.
- `module_paths` optionally append user module directories that override the built-ins.
- `rules` and `train_rules` define the classification/train flows; per-account overrides live under each `maildirs` entry.
- `categories` map folders to behaviours via `flag` (`ham`, `spam`, or `neutral`).

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

The status command inspects configured accounts, module state, PID liveness, and training metadata.

## CLI Reference

| Command | Description |
| --- | --- |
| `papaya daemon [-d]` | Run the daemon in foreground or background. |
| `papaya status` | Show configuration summary and whether the daemon is running. |
| `papaya classify path/to/message.eml [-a account]` | Execute the configured rules for a single message. |
| `papaya train [--full] [-a account]` | Replay categorised mail for manual training; `--full` clears trained-ID cache first and reloads modules with fresh models. |

## Development

Papaya ships with Ruff, mypy, and pytest. Use the Just recipes to keep everything consistent:

```bash
just fix   # format + Ruff autofix
just check # format check, lint, mypy, pytest
```

Tests live under `tests/unit`. Integration/e2e suites will plug in once maildir fixtures are available. The documentation in `docs/prd.md` and `docs/plan.md` covers the full roadmap through Phase 10; optimisation, deep health monitoring, and hardening are intentionally deferred for this milestone.

---

Need help or spot a bug? File an issue with your config snippet and relevant log excerpts from `<root_dir>/logs/`.
