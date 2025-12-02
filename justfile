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

# Execute pytest suite
test:
    uv run pytest

# Run all checks (format, lint, typing, tests)
check: formatcheck lint typecheck test

# Fix formatting and auto-fix Ruff issues (unsafe)
fix: format
    uv run ruff check --preview --fix --unsafe-fixes src tests
