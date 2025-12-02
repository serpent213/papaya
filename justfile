# Help (default)
help:
    @just --list

# Format code with Ruff
format:
    uv run ruff format src tests

# Check formatting without writing changes
formatcheck:
    uv run ruff format --check src tests

# Run Ruff lints
lint:
    uv run ruff check --preview src tests

# Run mypy type checker
typecheck:
    uv run mypy src

# Execute pytest suite (all by default, filtered variants available)
test kind="all":
    @case "{{kind}}" in \
        all) uv run pytest ;; \
        unit) uv run pytest -m "not integration" ;; \
        integration) uv run pytest -m integration ;; \
        slow) uv run pytest -m slow ;; \
        *) echo "Unknown test kind '{{kind}}'. Use all|unit|integration|slow." >&2; exit 1 ;; \
    esac

# Run all checks (format, lint, typing, tests)
check: formatcheck lint typecheck test

# Fix formatting and auto-fix Ruff issues (unsafe)
fix: format
    uv run ruff check --preview --fix --unsafe-fixes src tests
