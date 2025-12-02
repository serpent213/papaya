# Repository Guidelines

## Project Structure & Module Organization
Runtime code lives in `src/papaya`, grouped by domain: `modules/` for ML plugins, `rules/` for the rule engine helpers, and `services/` for watchers, stores, and IPC. CLI entrypoints are wired through `main.py`, while reusable docs and plans sit in `docs/` (see `ARCHITECTURE.md` for the high-level flow). Tests live in `tests/unit`, mirroring the source tree; add new fixtures beside the modules they exercise to keep context close.

## Build, Test, and Development Commands
Install deps once with `uv sync --all-extras`. Daily workflows rely on Just recipes:
```
just format      # Ruff formatter
just lint        # Ruff check --preview
just typecheck   # mypy src
just test        # pytest suite
just check       # formatcheck + lint + typecheck + test
just fix         # format + Ruff autofix (unsafe)
```
Use `uv run papaya daemon` to smoke-test the CLI against a sample config before opening a PR.

Always run `just fix` and `just test` WITHOUT sandbox after making changes (and in between as required).

## Coding Style & Naming Conventions
Target Python 3.10+, four-space indents, max line length 100, and double quotes (enforced via `ruff.toml`). Prefer explicit type annotations on public surfaces so `uv run mypy src` stays clean. Package names remain `snake_case`, classes use `PascalCase`, and async/background helpers keep an `_task` suffix for quick grepability.

## Testing Guidelines
Write pytest cases under `tests/unit/<module_name>/test_*.py` and keep assertions close to the behaviour under test (state-machine modules usually need both happy-path and failure-path coverage). When touching routers or daemon glue, include a regression test that mimics the relevant Maildir scenario; aim to extend fixtures instead of baking inline EML strings. Always run `just test` before pushing and update docs if new config knobs change expected behaviour.

## Commit & Pull Request Guidelines
Commits follow the lightweight conventional style visible in `git log` (`feat:`, `docs:`, `llm:`). Keep summaries under 72 chars and describe “why” in the body when context is non-obvious. PRs should link the motivating issue, summarise behavioural changes, note testing commands, and attach CLI output or screenshots when user-facing. Flag config or schema migrations clearly so reviewers can test against their own Maildir setups.

## Security & Configuration Tips
Never commit real Maildir paths, auth tokens, or personal sender lists. Use redacted samples inside `tests/fixtures` and document new knobs in `docs/prd.md` plus the README configuration block. When proposing new rules/modules, describe their failure modes and how to disable them via `~/.config/papaya/config.yaml` so operators can roll back confidently.
