"""Papaya command-line interface."""

from __future__ import annotations

import json
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
from .classifiers.naive_bayes import NaiveBayesClassifier
from .classifiers.registry import ClassifierRegistry
from .classifiers.tfidf_sgd import TfidfSgdClassifier
from .config import Config, ConfigError, MaildirAccount, load_config
from .dovecot import DovecotKeywords
from .extractor.features import extract_features
from .logging import configure_logging
from .maildir import read_message
from .mover import MailMover
from .pidfile import PidFile, PidFileError, pid_alive, read_pid
from .pipeline import Pipeline
from .runtime import AccountRuntime, DaemonRuntime
from .senders import SenderLists
from .store import Store
from .trainer import Trainer
from .types import Category, ClassifierMode, Prediction
from .watcher import MaildirWatcher

app = typer.Typer(help="Papaya spam daemon utilities.")
DEFAULT_PID_NAME = "papaya.pid"
LOGGER = logging.getLogger(__name__)


@dataclass
class CLIState:
    """Stores shared CLI options."""

    config_path: Path | None


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
) -> None:
    """Capture global CLI options."""

    resolved = config.expanduser() if config else None
    ctx.obj = CLIState(config_path=resolved)


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
        )
        return

    with PidFile(pid_path):
        try:
            _run_daemon_process(
                initial_config=config,
                config_path=state.config_path,
                skip_initial_training=skip_initial_training,
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
    config, store, _senders = _load_environment(state)
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
        for classifier_cfg in config.classifiers:
            model_path = store.model_path(classifier_cfg.name, account=account.name)
            exists = "present" if model_path.exists() else "missing"
            typer.echo(
                f"      {classifier_cfg.name} [{classifier_cfg.mode.name.lower()}]: {exists}"
            )
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
    """Classify a single RFC822 message using all configured classifiers."""

    state = _state(ctx)
    config, store, _senders = _load_environment(state)
    target = _resolve_account(config, account)
    try:
        registry = _build_registry(config, store, target.name, load_models=True)
    except ConfigError as exc:
        _config_failure(exc)
    message_path = message.expanduser()
    if not message_path.is_file():
        typer.secho(f"Message file not found: {message_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    try:
        parsed = read_message(message_path)
    except Exception as exc:
        typer.secho(f"Failed to parse message: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc

    features = extract_features(parsed)
    predictions = registry.predict_all(features)
    message_id = parsed.get("Message-ID") or message_path.name
    store.log_predictions(target.name, message_id, predictions)

    typer.echo(f"Message: {message_path}")
    typer.echo(f"Account: {target.name}")
    typer.echo(f"Message-ID: {message_id}")
    typer.echo("")
    _print_predictions(predictions, registry.modes())


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
    config, store, senders = _load_environment(state)
    targets = _select_accounts(config, [account] if account else None)

    if full:
        store.trained_ids.reset()

    for acct in targets:
        try:
            registry = _build_registry(config, store, acct.name, load_models=not full)
        except ConfigError as exc:
            _config_failure(exc)
        trainer = Trainer(
            account=acct.name,
            maildir=acct.path,
            registry=registry,
            senders=senders,
            store=store,
            categories=config.categories,
        )
        results = trainer.initial_training()
        trained = sum(1 for result in results if result.status == "trained")
        typer.echo(
            f"{acct.name}: processed {len(results)} message(s), trained {trained} new sample(s)."
        )


@app.command()
def compare(ctx: typer.Context) -> None:
    """Compare classifier accuracy using prediction logs and training events."""

    state = _state(ctx)
    config, store, _senders = _load_environment(state)
    log_path = store.prediction_log_path
    if not log_path.exists():
        typer.secho(f"Prediction log not found: {log_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    truth = store.trained_ids.snapshot()
    if not truth:
        typer.secho("No training data recorded yet.", fg=typer.colors.YELLOW)
        raise typer.Exit(1) from None

    stats: dict[str, tuple[int, int]] = {}
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            message_id = record.get("message_id")
            classifier_name = record.get("classifier")
            predicted = record.get("category")
            actual = truth.get(message_id)
            if not message_id or actual is None or classifier_name is None:
                continue
            total, correct = stats.get(classifier_name, (0, 0))
            total += 1
            if predicted == actual:
                correct += 1
            stats[classifier_name] = (total, correct)

    if not stats:
        typer.secho(
            "No overlapping predictions and training records found.", fg=typer.colors.YELLOW
        )
        raise typer.Exit(1) from None

    typer.echo("Classifier comparison:")
    for name, (total, correct) in sorted(stats.items()):
        accuracy = correct / total if total else 0.0
        typer.echo(f"  {name}: {correct}/{total} correct ({accuracy:.1%})")


def _state(ctx: typer.Context) -> CLIState:
    state = ctx.obj
    if not isinstance(state, CLIState):
        raise RuntimeError("CLI state missing from context.")
    return state


def _load_environment(state: CLIState) -> tuple[Config, Store, SenderLists]:
    config = _load_config(state.config_path)
    configure_logging(config.logging, config.root_dir)
    store = Store(config.root_dir)
    senders = SenderLists(config.root_dir)
    return config, store, senders


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
) -> None:
    config = initial_config
    configure_logging(config.logging, config.root_dir)
    store = Store(config.root_dir)
    senders = SenderLists(config.root_dir)

    def rebuild_accounts(load_models: bool = True) -> list[AccountRuntime]:
        return [
            _build_account_runtime(
                account,
                config,
                store,
                senders,
                load_models=load_models,
            )
            for account in config.maildirs
        ]

    def reload_callback() -> tuple[list[AccountRuntime], bool]:
        nonlocal config, store, senders
        try:
            new_config = load_config(config_path)
        except ConfigError as exc:  # pragma: no cover - exercised in reload tests
            LOGGER.error("Failed to reload Papaya configuration: %s", exc)
            raise
        if new_config.root_dir != config.root_dir:
            LOGGER.info(
                "Papaya root dir changed from %s to %s",
                config.root_dir,
                new_config.root_dir,
            )
            store = Store(new_config.root_dir)
            senders = SenderLists(new_config.root_dir)
        config = new_config
        configure_logging(config.logging, config.root_dir)
        return rebuild_accounts(load_models=True), True

    def status_callback(snapshots: list[dict[str, Any]]) -> str | None:
        return _format_status_message(config, snapshots)

    runtime = DaemonRuntime(
        rebuild_accounts(),
        reload_callback=reload_callback,
        status_callback=status_callback,
    )
    runtime.run(initial_training=not skip_initial_training)


def _build_daemon_runtime(config: Config, store: Store, senders: SenderLists) -> DaemonRuntime:
    accounts = [
        _build_account_runtime(account, config, store, senders) for account in config.maildirs
    ]
    return DaemonRuntime(accounts)


def _build_account_runtime(
    account: MaildirAccount,
    config: Config,
    store: Store,
    senders: SenderLists,
    *,
    load_models: bool = True,
) -> AccountRuntime:
    keywords = DovecotKeywords(account.path)
    try:
        papaya_flag = keywords.ensure_keyword()
    except RuntimeError as exc:  # pragma: no cover - configuration/environment error
        raise ConfigError(
            f"Failed to register Papaya keyword for account '{account.name}': {exc}"
        ) from exc
    mover = MailMover(account.path, papaya_flag=papaya_flag)
    registry = _build_registry(config, store, account.name, load_models=load_models)
    pipeline = Pipeline(
        account=account.name,
        maildir=account.path,
        registry=registry,
        senders=senders,
        store=store,
        categories=config.categories,
        mover=mover,
    )
    trainer = Trainer(
        account=account.name,
        maildir=account.path,
        registry=registry,
        senders=senders,
        store=store,
        categories=config.categories,
    )
    watcher = MaildirWatcher(account.path, config.categories.keys())
    return AccountRuntime(
        name=account.name,
        maildir=account.path,
        pipeline=pipeline,
        trainer=trainer,
        watcher=watcher,
        papaya_flag=papaya_flag,
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


CLASSIFIER_FACTORIES = {
    "naive_bayes": NaiveBayesClassifier,
    "tfidf_sgd": TfidfSgdClassifier,
}


def _build_registry(
    config: Config,
    store: Store,
    account_name: str,
    *,
    load_models: bool,
) -> ClassifierRegistry:
    registry = ClassifierRegistry()
    for classifier_cfg in config.classifiers:
        factory = CLASSIFIER_FACTORIES.get(classifier_cfg.type.lower())
        if factory is None:
            raise ConfigError(f"Unknown classifier type: {classifier_cfg.type}")
        classifier = factory(name=classifier_cfg.name)
        if load_models:
            store.load_classifier(classifier, account=account_name)
        registry.register(classifier, classifier_cfg.mode)
    return registry


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


def _print_predictions(
    predictions: dict[str, Prediction], modes: dict[str, ClassifierMode]
) -> None:
    if not predictions:
        typer.secho("No classifiers configured.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None
    name_width = max(len(name) for name in predictions)
    typer.echo("Predictions:")
    for name, prediction in predictions.items():
        mode = modes.get(name, ClassifierMode.SHADOW)
        flag = "A" if mode is ClassifierMode.ACTIVE else "S"
        category = _format_category(prediction.category)
        typer.echo(f"  {name:<{name_width}} [{flag}]: {category} ({prediction.confidence:.2%})")
    active_name = _active_classifier_name(modes)
    active_prediction = predictions.get(active_name)
    if active_prediction:
        typer.echo("─" * (name_width + 32))
        typer.echo(
            f"Consensus ({active_name}): {_format_category(active_prediction.category)} "
            f"({active_prediction.confidence:.2%})"
        )


def _format_category(category: Category | str | None) -> str:
    if isinstance(category, Category):
        return category.value
    if isinstance(category, str) and category:
        return category
    return "Inbox"


def _active_classifier_name(modes: dict[str, ClassifierMode]) -> str:
    for name, mode in modes.items():
        if mode is ClassifierMode.ACTIVE:
            return name
    raise RuntimeError("No active classifier configured.")


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
) -> None:
    process = multiprocessing.Process(
        target=_daemon_subprocess,
        args=(
            str(config_path) if config_path else None,
            str(pid_path),
            skip_initial_training,
            initial_config,
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
