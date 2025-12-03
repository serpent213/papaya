"""Papaya command-line interface."""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from . import __version__
from .config import Config, ConfigError, MaildirAccount, load_config
from .dovecot import DovecotKeywords
from .logging import configure_logging
from .maildir import read_message
from .modules import ModuleContext, ModuleLoader
from .mover import MailMover
from .pidfile import PidFile, PidFileError, pid_alive, read_pid
from .rules import RuleEngine, RuleError
from .runtime import AccountRuntime, DaemonRuntime
from .store import Store
from .trainer import Trainer
from .watcher import MaildirWatcher

app = typer.Typer(help="Papaya spam daemon utilities.")
DEFAULT_PID_NAME = "papaya.pid"
LOGGER = logging.getLogger(__name__)


@dataclass
class CLIState:
    """Stores shared CLI options."""

    config_path: Path | None
    dry_run: bool = False


@app.callback()
def _papaya(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "-c",
            "--config",
            help="Path to Papaya config (env PAPAYA_CONFIG or ~/.config/papaya/config.yaml).",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Log actions without modifying maildirs (except watcher directory prep).",
        ),
    ] = False,
) -> None:
    """Capture global CLI options."""

    resolved = config.expanduser() if config else None
    ctx.obj = CLIState(config_path=resolved, dry_run=dry_run)


@app.command()
def daemon(
    ctx: typer.Context,
    background: Annotated[
        bool,
        typer.Option(
            "-d",
            "--background",
            help="Run Papaya as a background process.",
        ),
    ] = False,
    pid_file: Annotated[
        Path | None,
        typer.Option(
            "--pid-file",
            help="Override PID file location (defaults to <root>/papaya.pid).",
        ),
    ] = None,
    skip_initial_training: Annotated[
        bool,
        typer.Option(
            "--skip-initial-training",
            help="Start daemon without replaying existing categorised mail.",
        ),
    ] = False,
) -> None:
    """Start the Papaya daemon."""

    state = _state(ctx)
    config = _load_config(state.config_path)
    pid_path = _pid_file_path(pid_file, config)
    try:
        PidFile.ensure_can_start(pid_path)
    except PidFileError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    if background:
        _start_background_daemon(
            initial_config=config,
            config_path=state.config_path,
            pid_path=pid_path,
            skip_initial_training=skip_initial_training,
            dry_run=state.dry_run,
        )
        return

    with PidFile(pid_path):
        try:
            _run_daemon_process(
                initial_config=config,
                config_path=state.config_path,
                skip_initial_training=skip_initial_training,
                dry_run=state.dry_run,
            )
        except ConfigError as exc:
            _config_failure(exc)


@app.command()
def status(
    ctx: typer.Context,
    pid_file: Annotated[
        Path | None,
        typer.Option(
            "--pid-file",
            help="Override PID file location used to detect running daemon.",
        ),
    ] = None,
) -> None:
    """Display Papaya configuration and daemon status."""

    state = _state(ctx)
    config, store = _load_environment(state)
    pid_path = _pid_file_path(pid_file, config)
    running, pid = _daemon_running(pid_path)

    typer.echo("→ Papaya Status")
    typer.echo(f"Version: {__version__}")
    typer.echo(f"Config path: {_resolved_config_path(state.config_path)}")
    typer.echo(f"Root dir: {config.root_dir}")
    daemon_line = "Daemon: ● Running" if running else "Daemon: ○ Stopped"
    if running and pid:
        daemon_line += f" (PID {pid})"
    typer.echo(daemon_line)
    typer.echo("")
    typer.echo("Accounts:")
    for account in config.maildirs:
        typer.echo(f"  - {account.name}: {account.path}")
    typer.echo("")
    typer.echo(f"Trained IDs: {len(store.trained_ids)}")
    typer.echo(f"Prediction log: {store.prediction_log_path}")


@app.command()
def classify(
    ctx: typer.Context,
    message: Annotated[Path, typer.Argument(..., help="Path to .eml message file.")],
    account: Annotated[
        str | None,
        typer.Option(
            "-a",
            "--account",
            help="Account name (defaults to first configured account).",
        ),
    ] = None,
) -> None:
    """Run the rule engine for a single RFC822 message."""

    state = _state(ctx)
    config, store = _load_environment(state)
    target = _resolve_account(config, account)
    message_path = message.expanduser()
    if not message_path.is_file():
        typer.secho(f"Message file not found: {message_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    try:
        parsed = read_message(message_path)
    except Exception as exc:
        typer.secho(f"Failed to parse message: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    module_loader, rule_engine = _initialise_rule_engine(config, store)
    try:
        try:
            message_id = (parsed.get("Message-ID") or "").strip() or message_path.name
            decision = rule_engine.execute_classify(
                target.name,
                parsed,
                message_id=message_id,
            )
        except RuleError as exc:
            typer.secho(f"Rule execution failed: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"Message: {message_path}")
        typer.echo(f"Account: {target.name}")
        typer.echo(f"Message-ID: {message_id}")
        typer.echo("Decision:")
        typer.echo(f"  action: {decision.action}")
        typer.echo(f"  category: {decision.category or 'inbox'}")
        confidence = f"{decision.confidence:.2f}" if decision.confidence is not None else "n/a"
        typer.echo(f"  confidence: {confidence}")
    finally:
        module_loader.call_cleanup()


@app.command()
def train(
    ctx: typer.Context,
    account: Annotated[
        str | None,
        typer.Option(
            "-a",
            "--account",
            help="Account to train (omit flag to process all accounts).",
        ),
    ] = None,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Retrain from scratch (clears trained ID registry).",
        ),
    ] = False,
) -> None:
    """Trigger manual training using categorised mail."""

    state = _state(ctx)
    config, store = _load_environment(state)
    targets = _select_accounts(config, [account] if account else None)

    if full:
        store.trained_ids.reset()

    module_loader, rule_engine = _initialise_rule_engine(config, store, fresh_models=full)
    try:
        for acct in targets:
            trainer = Trainer(
                account=acct.name,
                maildir=acct.path,
                store=store,
                categories=config.categories,
                rule_engine=rule_engine,
            )
            results = trainer.initial_training()
            trained = sum(1 for result in results if result.status == "trained")
            typer.echo(
                f"{acct.name}: processed {len(results)} message(s), "
                f"trained {trained} new sample(s)."
            )
    finally:
        module_loader.call_cleanup()


def _state(ctx: typer.Context) -> CLIState:
    state = ctx.obj
    if not isinstance(state, CLIState):
        raise RuntimeError("CLI state missing from context.")
    return state


def _load_environment(state: CLIState) -> tuple[Config, Store]:
    config = _load_config(state.config_path)
    configure_logging(config.logging, config.root_dir)
    store = Store(config.root_dir)
    return config, store


def _load_config(path: Path | None) -> Config:
    try:
        return load_config(path)
    except ConfigError as exc:  # pragma: no cover - exercised via CLI tests
        _config_failure(exc)


def _config_failure(exc: ConfigError) -> NoReturn:
    typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(2) from exc


def _run_daemon_process(
    *,
    initial_config: Config,
    config_path: Path | None,
    skip_initial_training: bool,
    dry_run: bool,
) -> None:
    config = initial_config
    configure_logging(config.logging, config.root_dir)
    store = Store(config.root_dir)
    module_loader, rule_engine = _initialise_rule_engine(config, store)

    def _build_accounts(
        engine: RuleEngine,
        cfg: Config,
        store_obj: Store,
    ) -> list[AccountRuntime]:
        return [
            _build_account_runtime(
                account,
                cfg,
                store_obj,
                engine,
                dry_run=dry_run,
            )
            for account in cfg.maildirs
        ]

    def rebuild_accounts() -> list[AccountRuntime]:
        return _build_accounts(rule_engine, config, store)

    def reload_callback() -> tuple[list[AccountRuntime], bool]:
        nonlocal config, store, module_loader, rule_engine
        try:
            new_config = load_config(config_path)
        except ConfigError as exc:  # pragma: no cover - exercised in reload tests
            LOGGER.error("Failed to reload Papaya configuration: %s", exc)
            raise
        new_store = store
        if new_config.root_dir != config.root_dir:
            LOGGER.info(
                "Papaya root dir changed from %s to %s",
                config.root_dir,
                new_config.root_dir,
            )
            new_store = Store(new_config.root_dir)
        new_loader, new_rule_engine = _initialise_rule_engine(new_config, new_store)
        try:
            new_accounts = _build_accounts(
                new_rule_engine,
                new_config,
                new_store,
            )
        except Exception:
            new_loader.call_cleanup()
            raise
        module_loader.call_cleanup()
        module_loader = new_loader
        rule_engine = new_rule_engine
        config = new_config
        store = new_store
        configure_logging(config.logging, config.root_dir)
        return new_accounts, True

    def status_callback(snapshots: list[dict[str, Any]]) -> str | None:
        return _format_status_message(config, snapshots)

    runtime = DaemonRuntime(
        rebuild_accounts(),
        reload_callback=reload_callback,
        status_callback=status_callback,
    )
    try:
        runtime.run(initial_training=not skip_initial_training)
    finally:
        module_loader.call_cleanup()


def _build_account_runtime(
    account: MaildirAccount,
    config: Config,
    store: Store,
    rule_engine: RuleEngine,
    *,
    dry_run: bool = False,
) -> AccountRuntime:
    keywords = DovecotKeywords(account.path)
    if dry_run:
        papaya_flag = keywords.existing_letter()
    else:
        try:
            papaya_flag = keywords.ensure_keyword()
        except RuntimeError as exc:  # pragma: no cover - configuration/environment error
            raise ConfigError(
                f"Failed to register Papaya keyword for account '{account.name}': {exc}"
            ) from exc
    mover = MailMover(account.path, papaya_flag=papaya_flag, dry_run=dry_run)
    trainer = Trainer(
        account=account.name,
        maildir=account.path,
        store=store,
        categories=config.categories,
        rule_engine=rule_engine,
    )
    watcher = MaildirWatcher(account.path, config.categories.keys())
    return AccountRuntime(
        name=account.name,
        maildir=account.path,
        rule_engine=rule_engine,
        mover=mover,
        trainer=trainer,
        watcher=watcher,
        categories=config.categories,
        papaya_flag=papaya_flag,
        dry_run=dry_run,
    )


def _format_status_message(config: Config, snapshots: list[dict[str, Any]]) -> str:
    root = config.root_dir
    if not snapshots:
        return f"Papaya daemon: no configured accounts (root={root})."
    lines = [f"Papaya daemon (root={root}) status:"]
    for snapshot in snapshots:
        name = snapshot["account"]
        watcher = "running" if snapshot["watcher_running"] else "stopped"
        processed = snapshot["processed"]
        inbox = snapshot["inbox_deliveries"]
        category_summary = ", ".join(
            f"{category}={count}"
            for category, count in sorted(snapshot["category_deliveries"].items())
        )
        if not category_summary:
            category_summary = "no_moves"
        lines.append(
            f"  - {name}: watcher={watcher} processed={processed} inbox={inbox} {category_summary}"
        )
    return "\n".join(lines)


def _module_search_paths(config: Config) -> list[Path]:
    builtin = (Path(__file__).resolve().parent / "modules").resolve()
    if not builtin.exists():
        raise ConfigError(f"Built-in module directory missing: {builtin}")
    return [builtin, *config.module_paths]


def _initialise_rule_engine(
    config: Config,
    store: Store,
    *,
    fresh_models: bool = False,
) -> tuple[ModuleLoader, RuleEngine]:
    loader = ModuleLoader(_module_search_paths(config))
    loader.load_all()
    context = ModuleContext(config=config, store=store, fresh_models=fresh_models)
    loader.call_startup(context)
    engine = RuleEngine(loader, store, config.rules, config.train_rules)
    for account in config.maildirs:
        engine.set_account_rules(account.name, account.rules, account.train_rules)
    return loader, engine


def _resolve_account(config: Config, name: str | None) -> MaildirAccount:
    if name is None:
        return config.maildirs[0]
    for account in config.maildirs:
        if account.name == name:
            return account
    typer.secho(f"Unknown account '{name}'.", fg=typer.colors.RED, err=True)
    raise typer.Exit(1) from None


def _select_accounts(config: Config, requested: Iterable[str] | None) -> list[MaildirAccount]:
    if not requested:
        return list(config.maildirs)
    accounts = []
    known = {account.name: account for account in config.maildirs}
    for name in requested:
        try:
            accounts.append(known[name])
        except KeyError:
            typer.secho(f"Unknown account '{name}'.", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
    return accounts


def _pid_file_path(pid_file: Path | None, config: Config) -> Path:
    if pid_file:
        return pid_file.expanduser()
    return (config.root_dir / DEFAULT_PID_NAME).expanduser()


def _daemon_running(pid_path: Path) -> tuple[bool, int | None]:
    pid = read_pid(pid_path)
    if pid is None:
        return False, None
    if pid_alive(pid):
        return True, pid
    pid_path.unlink(missing_ok=True)
    return False, None


def _start_background_daemon(
    *,
    initial_config: Config,
    config_path: Path | None,
    pid_path: Path,
    skip_initial_training: bool,
    dry_run: bool,
) -> None:
    process = multiprocessing.Process(
        target=_daemon_subprocess,
        args=(
            str(config_path) if config_path else None,
            str(pid_path),
            skip_initial_training,
            initial_config,
            dry_run,
        ),
        daemon=False,
    )
    process.start()
    typer.echo(f"Papaya daemon started in background (PID {process.pid}).")


def _daemon_subprocess(
    config_path_str: str | None,
    pid_path_str: str,
    skip_initial_training: bool,
    initial_config: Config | None = None,
    dry_run: bool = False,
) -> None:
    config_path = Path(config_path_str).expanduser() if config_path_str else None
    pid_path = Path(pid_path_str)
    pidfile = PidFile(pid_path)
    try:
        pidfile.create()
    except PidFileError as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"{exc}\n")
        return
    try:
        if initial_config is None:
            config = load_config(config_path)
        else:
            config = initial_config
    except ConfigError as exc:  # pragma: no cover - child process logging
        sys.stderr.write(f"Papaya configuration error: {exc}\n")
        pidfile.remove()
        return
    try:
        _run_daemon_process(
            initial_config=config,
            config_path=config_path,
            skip_initial_training=skip_initial_training,
            dry_run=dry_run,
        )
    except ConfigError as exc:  # pragma: no cover - background path
        sys.stderr.write(f"Papaya configuration error: {exc}\n")
    finally:
        pidfile.remove()


def _resolved_config_path(path: Path | None) -> Path:
    if path:
        return path
    env = os.environ.get("PAPAYA_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path("~/.config/papaya/config.yaml").expanduser()


def main() -> None:  # pragma: no cover - delegated to Typer
    app()


__all__ = ["app", "main"]
