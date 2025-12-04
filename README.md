Papaya Spam Eater
==================

**Personal/small-scale post-SMTP mail sorter**

Papaya watches maildirs on filesystem level, acting on incoming mail and sorting it into predefined folders. It acknowledges the need for a flexible design that invites change and regular optimisation by providing a combination of hot-pluggable extractor and classifier modules and global or per-account rulesets written in a Python-based DSL.

## Highlights

- **Rule-driven automation** – Python snippets embedded in config files coordinate module calls, sender shortcuts, and routing decisions.
- **Hot-reloadable modules** – Built-in and user modules expose classify/train hooks with persisted state managed through the store.
- **Sender memory** – The `match_from` module remembers per-account sender/category pairs so known newsletters or VIPs skip ML entirely.
- **Continuous learning** – Every user correction triggers immediate training through the `train` block; module state is persisted incrementally.
- **Modern CLI** – Typer-based commands for daemon control, manual classification, and bulk training.
- **Runtime control** – Foreground/background modes with PID management, SIGHUP config reloads, and SIGUSR1 status dumps.

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
  known_category = modules.match_from.classify(message, None, account)
  if known_category:
      move_to(known_category)
  else:
      features = modules.extract_features.classify(message)
      prediction = modules.naive_bayes.classify(message, features, account)

      if prediction.category and prediction.confidence >= 0.55:
          move_to(prediction.category, confidence=prediction.confidence)
      else:
          skip()

train: |
  features = modules.extract_features.classify(message)
  modules.naive_bayes.train(message, features, category, account)
  modules.tfidf_sgd.train(message, features, category, account)
  modules.match_from.train(message, features, category, account)

categories:
  Spam: {}
  Newsletters: {}
  Important: {}
logging:
  level: info
  write_debug_logfile: false
  write_predictions_logfile: true
```

- `root_dir` stores classifier models, the `match_from` cache, logs, and the PID file.
- `module_paths` optionally append user module directories that override the built-ins.
- `rules` and `train` define the classification/train flows; per-account overrides live under each `maildirs` entry. The defaults check `match_from` before invoking ML.
- `categories` list the Maildir folders Papaya should watch/train; keys must match the on-disk directory names.
- `logging.write_debug_logfile` enables `logs/debug.log`, while `logging.write_predictions_logfile` controls the structured `logs/predictions.log`.

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
- `--dry-run` – exercise the daemon without moving/rewriting messages (watchers still create folders).

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
| `papaya train [--full] [-a account]` | Replay categorised mail for manual training; `--full` clears trained-ID cache first and reloads modules with reset state. |

## Development

Papaya ships with Ruff, mypy, and pytest. Use the Just recipes to keep everything consistent:

```bash
just fix   # format + Ruff autofix
just check # format check, lint, mypy, pytest
```

Tests live under `tests/unit`. Integration/e2e suites will plug in once maildir fixtures are available. The documentation in `docs/prd.md` and `docs/plan.md` covers the full roadmap through Phase 10; optimisation, deep health monitoring, and hardening are intentionally deferred for this milestone.

---

Need help or spot a bug? File an issue with your config snippet and relevant log excerpts from `<root_dir>/logs/`.
