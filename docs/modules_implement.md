# Papaya Module System Implementation Guide

## Overview

This document details the implementation of Papaya's rule engine refactor, replacing the rigid pipeline with a flexible, user-configurable system. Three major components:

1. **Module System** — Hot-reloadable Python modules with lifecycle hooks
2. **Rule DSL** — Python snippets embedded in YAML orchestrating modules
3. **Dovecot Keyword Integration** — Persistent flag marking daemon-sorted mail

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DaemonRuntime                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                           ModuleLoader                                   ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    ││
│  │  │ extract_    │  │ naive_      │  │ tfidf_      │  │ user_       │    ││
│  │  │ features    │  │ bayes       │  │ sgd         │  │ custom      │    ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                            RuleEngine                                    ││
│  │  ┌─────────────────────┐  ┌─────────────────────┐                       ││
│  │  │ Global Rules        │  │ Account Rules       │                       ││
│  │  │ (compiled)          │  │ (compiled, cached)  │                       ││
│  │  └─────────────────────┘  └─────────────────────┘                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         ▼                          ▼                          ▼             │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │ Account     │           │ Account     │           │ Account     │       │
│  │ Runtime     │           │ Runtime     │           │ Runtime     │       │
│  │ (personal)  │           │ (work)      │           │ (shared)    │       │
│  └─────────────┘           └─────────────┘           └─────────────┘       │
│         │                          │                          │             │
│         ▼                          ▼                          ▼             │
│  ┌─────────────┐           ┌─────────────┐           ┌─────────────┐       │
│  │ Dovecot     │           │ Dovecot     │           │ Dovecot     │       │
│  │ Keywords    │           │ Keywords    │           │ Keywords    │       │
│  └─────────────┘           └─────────────┘           └─────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Module System

### 1.1 Directory Structure

```
src/papaya/modules/              # Built-in modules (shipped with package)
├── __init__.py                  # Module system exports
├── loader.py                    # ModuleLoader implementation
├── context.py                   # ModuleContext dataclass
├── extract_features.py          # Wraps extractor/features.py
├── naive_bayes.py               # Wraps classifiers/naive_bayes.py
└── tfidf_sgd.py                 # Wraps classifiers/tfidf_sgd.py

~/.config/papaya/modules/        # User modules (optional)
└── custom_filter.py             # User-defined module
```

### 1.2 Module Interface

Each module is a Python file or package. All hooks are **optional** — only implement what you need.

```python
"""Example module: modules/naive_bayes.py"""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext
    from papaya.types import Features

# Module-level state (initialised in startup)
_models: dict[str, NaiveBayesClassifier] = {}
_store: Store | None = None


def startup(ctx: ModuleContext) -> None:
    """Called once when daemon starts. Load models, initialise state.

    Args:
        ctx: Provides access to Store, Config, and logging.
    """
    global _models, _store
    _store = ctx.store

    for account in ctx.config.maildirs:
        classifier = NaiveBayesClassifier(name="naive_bayes")
        ctx.store.load_classifier(classifier, account=account.name)
        _models[account.name] = classifier


def classify(message: EmailMessage, features: Features, account: str) -> Prediction:
    """Called during mail classification.

    Args:
        message: Parsed email message.
        features: Extracted features (from extract_features module).
        account: Account name for model selection.

    Returns:
        Prediction with category, confidence, and scores dict.
    """
    model = _models.get(account)
    if model is None or not model.is_trained():
        return Prediction(category=None, confidence=0.0, scores={})
    return model.predict(features)


def train(message: EmailMessage, features: Features, category: str, account: str) -> None:
    """Called when user sorts mail (training signal).

    Args:
        message: Parsed email message.
        features: Extracted features.
        category: Target category (where user moved the mail).
        account: Account name.
    """
    model = _models.get(account)
    if model is None:
        return

    label = Category(category)  # Convert string to enum
    model.train(features, label)

    if _store:
        _store.save_classifier(model, account=account)


def cleanup() -> None:
    """Called on daemon shutdown. Persist final state, release resources."""
    # Models auto-saved in train(), nothing else needed
    _models.clear()
```

### 1.3 ModuleContext

```python
# src/papaya/modules/context.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papaya.config import Config
    from papaya.store import Store


@dataclass(frozen=True)
class ModuleContext:
    """Context passed to module startup() hooks."""

    config: Config
    store: Store

    # Future extensions:
    # logger: logging.Logger
    # senders: SenderLists
```

### 1.4 ModuleLoader

```python
# src/papaya/modules/loader.py

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext

LOGGER = logging.getLogger(__name__)


class ModuleLoader:
    """Discovers, loads, and manages Papaya modules."""

    def __init__(self, paths: list[Path]) -> None:
        """Initialise loader with search paths.

        Args:
            paths: Directories to search for modules. Later paths override
                   earlier ones (user modules override built-ins).
        """
        self._paths = [p.expanduser() for p in paths]
        self._modules: dict[str, ModuleType] = {}
        self._load_order: list[str] = []  # For deterministic startup/cleanup

    def load_all(self) -> None:
        """Discover and import all modules from configured paths."""
        discovered: dict[str, Path] = {}

        for search_path in self._paths:
            if not search_path.exists():
                continue

            for candidate in search_path.iterdir():
                name = self._module_name(candidate)
                if name and name not in ("__init__", "loader", "context"):
                    discovered[name] = candidate  # Later paths override

        for name, path in discovered.items():
            try:
                module = self._import_module(name, path)
                self._modules[name] = module
                self._load_order.append(name)
                LOGGER.debug("Loaded module '%s' from %s", name, path)
            except Exception:
                LOGGER.exception("Failed to load module '%s' from %s", name, path)
                raise

    def reload_all(self) -> None:
        """Hot-reload all modules. Called on SIGHUP."""
        # Clear existing modules
        for name in self._load_order:
            if name in sys.modules:
                # Remove from sys.modules to force reimport
                full_name = f"papaya.modules.{name}"
                sys.modules.pop(full_name, None)
                sys.modules.pop(name, None)

        self._modules.clear()
        self._load_order.clear()

        # Reload
        self.load_all()

    def get(self, name: str) -> ModuleType:
        """Get module by name.

        Raises:
            KeyError: If module not found.
        """
        if name not in self._modules:
            raise KeyError(f"Module '{name}' not found. Available: {list(self._modules.keys())}")
        return self._modules[name]

    def call_startup(self, ctx: ModuleContext) -> None:
        """Call startup() on all modules that define it."""
        for name in self._load_order:
            module = self._modules[name]
            if hasattr(module, "startup"):
                LOGGER.debug("Calling startup() on module '%s'", name)
                try:
                    module.startup(ctx)
                except Exception:
                    LOGGER.exception("Module '%s' startup() failed", name)
                    raise

    def call_cleanup(self) -> None:
        """Call cleanup() on all modules in reverse order."""
        for name in reversed(self._load_order):
            module = self._modules[name]
            if hasattr(module, "cleanup"):
                LOGGER.debug("Calling cleanup() on module '%s'", name)
                try:
                    module.cleanup()
                except Exception:
                    LOGGER.exception("Module '%s' cleanup() failed", name)

    @property
    def module_names(self) -> list[str]:
        """Return list of loaded module names."""
        return list(self._load_order)

    def _module_name(self, path: Path) -> str | None:
        """Extract module name from path, or None if not a valid module."""
        if path.is_file() and path.suffix == ".py":
            return path.stem
        if path.is_dir() and (path / "__init__.py").exists():
            return path.name
        return None

    def _import_module(self, name: str, path: Path) -> ModuleType:
        """Import a module from filesystem path."""
        if path.is_dir():
            module_path = path / "__init__.py"
        else:
            module_path = path

        spec = importlib.util.spec_from_file_location(
            f"papaya.modules.{name}",
            module_path,
            submodule_search_locations=[str(path.parent)] if path.is_dir() else None,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
```

### 1.5 Built-in Module: extract_features

```python
# src/papaya/modules/extract_features.py

"""Feature extraction module wrapping the existing extractor."""

from __future__ import annotations

from email.message import EmailMessage
from typing import TYPE_CHECKING

from papaya.extractor.features import extract_features as _extract

if TYPE_CHECKING:
    from papaya.modules.context import ModuleContext
    from papaya.types import Features


def startup(ctx: ModuleContext) -> None:
    """No initialisation needed for stateless extractor."""
    pass


def classify(message: EmailMessage) -> Features:
    """Extract features from email message.

    Note: This module doesn't use 'account' - features are account-agnostic.
    """
    return _extract(message)


# No train() - feature extraction doesn't learn
# No cleanup() - stateless
```

---

## 2. Rule DSL Engine

### 2.1 Rule Execution Model

Rules are Python code snippets executed via `exec()`. The execution namespace provides:

| Name | Type | Description |
|------|------|-------------|
| `message` | `EmailMessage` | Parsed email being processed |
| `account` | `str` | Current account name |
| `category` | `str \| None` | Target category (train context only) |
| `modules` | `ModuleNamespace` | Access to loaded modules |
| `move_to(cat, confidence=1.0)` | function | Declare move decision |
| `skip()` | function | Declare inbox decision |
| `fallback()` | function | Defer to global rules |

### 2.2 RuleDecision

```python
# src/papaya/rules.py

from __future__ import annotations

from dataclasses import dataclass
from types import CodeType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from email.message import EmailMessage
    from papaya.modules.loader import ModuleLoader


@dataclass(frozen=True)
class RuleDecision:
    """Result of rule execution."""

    action: str  # "move" | "inbox" | "fallback"
    category: str | None = None
    confidence: float | None = None


class RuleError(Exception):
    """Raised when rule execution fails."""

    def __init__(self, message: str, *, rule_source: str | None = None, cause: Exception | None = None):
        super().__init__(message)
        self.rule_source = rule_source
        self.cause = cause
```

### 2.3 ModuleNamespace

```python
class ModuleNamespace:
    """Provides attribute-style access to modules: modules.naive_bayes.classify(...)"""

    def __init__(self, loader: ModuleLoader) -> None:
        self._loader = loader

    def __getattr__(self, name: str):
        try:
            return self._loader.get(name)
        except KeyError:
            raise AttributeError(f"Module '{name}' not found") from None

    def __dir__(self):
        return self._loader.module_names
```

### 2.4 RuleEngine

```python
class RuleEngine:
    """Compiles and executes classification/training rules."""

    def __init__(
        self,
        loader: ModuleLoader,
        store: Store,
        global_rules: str,
        global_train_rules: str,
    ) -> None:
        self._loader = loader
        self._store = store
        self._global_rules = self._compile(global_rules, "<global_rules>")
        self._global_train_rules = self._compile(global_train_rules, "<global_train_rules>")
        self._account_rules: dict[str, CodeType] = {}
        self._account_train_rules: dict[str, CodeType] = {}

    def set_account_rules(
        self,
        account: str,
        rules: str | None,
        train_rules: str | None,
    ) -> None:
        """Compile and cache account-specific rules."""
        if rules:
            self._account_rules[account] = self._compile(rules, f"<{account}_rules>")
        if train_rules:
            self._account_train_rules[account] = self._compile(train_rules, f"<{account}_train_rules>")

    def execute_classify(
        self,
        account: str,
        message: EmailMessage,
        *,
        message_id: str,
    ) -> RuleDecision:
        """Execute classification rules for account.

        Flow:
        1. If account has custom rules, execute them
        2. If custom rules return fallback() or don't decide, run global rules
        3. If nothing decides, default to inbox
        """
        namespace = self._build_classify_namespace(message, account, message_id)

        # Try account-specific rules first
        if account in self._account_rules:
            self._exec_safe(self._account_rules[account], namespace, account)
            decision = namespace.get("_decision")
            if decision and decision.action != "fallback":
                return decision
            namespace["_decision"] = None

        # Fallback to global rules
        self._exec_safe(self._global_rules, namespace, "global")
        decision = namespace.get("_decision")

        return decision or RuleDecision(action="inbox")

    def execute_train(self, account: str, message: EmailMessage, category: str) -> None:
        """Execute training rules for account."""
        namespace = self._build_train_namespace(message, account, category)

        # Use account-specific train rules if available, else global
        rules = self._account_train_rules.get(account) or self._global_train_rules
        self._exec_safe(rules, namespace, account)

    def _build_classify_namespace(
        self,
        message: EmailMessage,
        account: str,
        message_id: str,
    ) -> dict:
        """Build execution namespace for classification."""
        decision_holder: dict[str, RuleDecision | None] = {"_decision": None}
        from_address = (message.get("From") or "").strip() or None
        subject = message.get("Subject")

        def move_to(category: str, confidence: float = 1.0) -> None:
            decision_holder["_decision"] = RuleDecision(
                action="move",
                category=category,
                confidence=confidence,
            )

        def skip() -> None:
            decision_holder["_decision"] = RuleDecision(action="inbox")

        def fallback() -> None:
            decision_holder["_decision"] = RuleDecision(action="fallback")

        def log(classifier: str, prediction: Prediction) -> None:
            """Append classifier scores plus message metadata to predictions.log."""
            self._store.log_predictions(
                account,
                message_id or "<missing>",
                {classifier: prediction},
                recipient=account,
                from_address=from_address,
                subject=subject,
            )

        return {
            "message": message,
            "account": account,
            "message_id": message_id,
            "modules": ModuleNamespace(self._loader),
            "move_to": move_to,
            "skip": skip,
            "fallback": fallback,
            "log": log,
            "_decision": None,
            "__builtins__": __builtins__,  # Allow standard Python
        }

    def _build_train_namespace(self, message: EmailMessage, account: str, category: str) -> dict:
        """Build execution namespace for training."""
        return {
            "message": message,
            "account": account,
            "category": category,
            "modules": ModuleNamespace(self._loader),
            "__builtins__": __builtins__,
        }

    def _compile(self, source: str, filename: str) -> CodeType:
        """Compile rule source to code object."""
        try:
            return compile(source, filename, "exec")
        except SyntaxError as e:
            raise RuleError(
                f"Syntax error in rules: {e.msg} at line {e.lineno}",
                rule_source=source,
                cause=e,
            ) from e

    def _exec_safe(self, code: CodeType, namespace: dict, context: str) -> None:
        """Execute compiled rules with error handling."""
        try:
            exec(code, namespace)
        except Exception as e:
            raise RuleError(
                f"Rule execution failed for {context}: {e}",
                cause=e,
            ) from e
```

---

## 3. Dovecot Keyword Integration

### 3.1 Keyword File Format

Dovecot stores keyword→letter mappings in `dovecot-keywords` per maildir:

```
0 $Junk
1 $NonJunk
2 $PapayaSorted
```

Index 0 → letter `a`, index 1 → letter `b`, etc.

Filenames include keywords as lowercase letters after standard flags:

```
1638371976.M123456.host:2,RSc
                         │││
                         ││└─ keyword 'c' = $PapayaSorted
                         │└── Seen
                         └─── Replied
```

### 3.2 DovecotKeywords Class

```python
# src/papaya/dovecot.py

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PAPAYA_KEYWORD = "$PapayaSorted"


class DovecotKeywordError(RuntimeError):
    """Raised when keyword registration fails."""


class DovecotKeywords:
    """Manage dovecot-keywords file for a maildir."""

    def __init__(self, maildir: Path) -> None:
        self._maildir = maildir.expanduser()
        self._path = self._maildir / "dovecot-keywords"
        self._letter: str | None = None

    def ensure_keyword(self) -> str:
        """Register $PapayaSorted keyword, return assigned letter (a-z).

        Creates dovecot-keywords file if it doesn't exist.
        Reuses existing slot if keyword already registered.

        Raises:
            DovecotKeywordError: If all 26 slots are taken.
        """
        existing = self._load()

        # Check if already registered
        for idx, name in existing.items():
            if name == PAPAYA_KEYWORD:
                self._letter = chr(ord("a") + idx)
                LOGGER.debug(
                    "Papaya keyword already registered as '%s' in %s",
                    self._letter,
                    self._path,
                )
                return self._letter

        # Find first free slot
        for idx in range(26):
            if idx not in existing:
                existing[idx] = PAPAYA_KEYWORD
                self._save(existing)
                self._letter = chr(ord("a") + idx)
                LOGGER.info(
                    "Registered Papaya keyword as '%s' in %s",
                    self._letter,
                    self._path,
                )
                return self._letter

        raise DovecotKeywordError(
            f"No free keyword slots in {self._path}. "
            "Remove unused keywords or use a different maildir."
        )

    @property
    def letter(self) -> str:
        """Return the assigned keyword letter.

        Raises:
            RuntimeError: If ensure_keyword() hasn't been called.
        """
        if self._letter is None:
            raise RuntimeError(
                "Keyword not initialised. Call ensure_keyword() first."
            )
        return self._letter

    @property
    def is_initialised(self) -> bool:
        """Return True if keyword has been registered."""
        return self._letter is not None

    def _load(self) -> dict[int, str]:
        """Parse dovecot-keywords file into {index: keyword_name}."""
        if not self._path.exists():
            return {}

        result: dict[int, str] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or " " not in line:
                continue
            idx_str, name = line.split(" ", 1)
            try:
                result[int(idx_str)] = name
            except ValueError:
                continue
        return result

    def _save(self, keywords: dict[int, str]) -> None:
        """Write dovecot-keywords file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{idx} {name}" for idx, name in sorted(keywords.items())]
        content = "\n".join(lines)
        if content:
            content += "\n"
        self._path.write_text(content, encoding="utf-8")
```

### 3.3 Maildir Flag Operations

```python
# src/papaya/maildir.py (additions)

def parse_maildir_info(filename: str) -> tuple[str, str, str]:
    """Parse maildir filename into (base, standard_flags, keyword_flags).

    Example: '123.host:2,RSab' → ('123.host', 'RS', 'ab')

    Standard flags are uppercase (D, F, P, R, S, T).
    Keyword flags are lowercase (a-z).
    """
    if ":2," not in filename:
        return filename, "", ""

    base, flags = filename.rsplit(":2,", 1)
    standard = "".join(c for c in flags if c.isupper())
    keywords = "".join(c for c in flags if c.islower())
    return base, standard, keywords


def build_maildir_filename(base: str, standard_flags: str, keyword_flags: str) -> str:
    """Reconstruct maildir filename from components.

    Flags are sorted: standard (uppercase) first, then keywords (lowercase).
    """
    all_flags = "".join(sorted(standard_flags)) + "".join(sorted(keyword_flags))
    return f"{base}:2,{all_flags}"


def add_keyword_flag(filename: str, letter: str) -> str:
    """Add a keyword flag letter to filename, maintaining sorted order."""
    base, standard, keywords = parse_maildir_info(filename)
    if letter in keywords:
        return filename
    keywords = "".join(sorted(set(keywords) | {letter}))
    return build_maildir_filename(base, standard, keywords)


def remove_keyword_flag(filename: str, letter: str) -> str:
    """Remove a keyword flag letter from filename."""
    base, standard, keywords = parse_maildir_info(filename)
    if letter not in keywords:
        return filename
    keywords = keywords.replace(letter, "")
    return build_maildir_filename(base, standard, keywords)


def has_keyword_flag(filename: str, letter: str) -> bool:
    """Check if filename has the given keyword flag letter."""
    _, _, keywords = parse_maildir_info(filename)
    return letter in keywords
```

### 3.4 Updated MailMover

```python
# src/papaya/mover.py (updated)

class MailMover:
    """Move messages between inbox and category folders."""

    def __init__(
        self,
        maildir: Path,
        *,
        hostname: str | None = None,
        papaya_flag: str | None = None,
    ) -> None:
        self._maildir = maildir.expanduser()
        self._hostname = (hostname or socket.gethostname() or "papaya").strip() or "papaya"
        self._papaya_flag = papaya_flag  # Letter from DovecotKeywords

    def move_to_category(
        self,
        msg_path: Path,
        category: str | Category,
        *,
        add_papaya_flag: bool = True,
    ) -> Path:
        """Move message to category folder, optionally marking as daemon-sorted."""
        destination = category_subdir(self._maildir, category, "cur")
        return self._move(
            Path(msg_path),
            destination,
            add_papaya_flag=add_papaya_flag and self._papaya_flag is not None,
        )

    def move_to_inbox(self, msg_path: Path) -> Path:
        """Move message to inbox cur/."""
        destination = inbox_cur_dir(self._maildir)
        return self._move(Path(msg_path), destination, add_papaya_flag=False)

    def _move(
        self,
        source: Path,
        destination_dir: Path,
        *,
        add_papaya_flag: bool = False,
    ) -> Path:
        # ... existing validation ...

        while True:
            new_name = self._generate_cur_name()

            # Add papaya flag if requested
            if add_papaya_flag and self._papaya_flag:
                new_name = add_keyword_flag(new_name, self._papaya_flag)

            candidate = destination_dir / new_name
            if candidate.exists():
                continue

            try:
                source.replace(candidate)
            except FileNotFoundError as exc:
                raise MaildirError(f"Message disappeared: {source}") from exc

            return candidate
```

---

## 4. Training Guard Integration

### 4.1 Enhanced AutoClassificationCache

The existing `AutoClassificationCache` in `runtime.py` handles the in-memory TTL cache. We enhance it to work with Dovecot flags:

```python
# src/papaya/runtime.py (updated)

@dataclass
class AccountRuntime:
    """Bundles all runtime objects for a single account."""

    name: str
    maildir: Path
    rule_engine: RuleEngine  # Replaces pipeline
    trainer: Trainer
    watcher: MaildirWatcher
    mover: MailMover
    dovecot_keywords: DovecotKeywords
    auto_cache: AutoClassificationCache = field(default_factory=AutoClassificationCache)

    def _handle_new_mail(self, path: Path) -> None:
        """Process new mail through rule engine."""
        message = read_message(path)
        message_id = self._extract_message_id(message, path)

        # Check sender lists (could also move into rules later)
        from_addr = self._extract_from(message)
        if self._handle_sender_shortcuts(path, from_addr, message_id):
            return

        # Execute classification rules
        decision = self.rule_engine.execute_classify(
            self.name,
            message,
            message_id=message_id,
        )

        if decision.action == "move" and decision.category:
            dest = self.mover.move_to_category(path, decision.category, add_papaya_flag=True)
            self.auto_cache.add(message_id)
            LOGGER.info("Classified %s → %s (confidence=%.2f)", message_id, decision.category, decision.confidence or 0)
        else:
            dest = self.mover.move_to_inbox(path)
            LOGGER.debug("Message %s → inbox", message_id)

    def _handle_user_sort(self, path: Path, category: str) -> None:
        """Process user-sorted mail for training."""
        message_id = extract_message_id(path)

        # Check 1: Was this just auto-classified? (in-memory cache)
        if message_id and self.auto_cache.consume(message_id):
            LOGGER.debug("Skipping training: daemon just classified %s", message_id)
            return

        # Check 2: Does it have papaya flag? (persistent marker)
        papaya_letter = self.dovecot_keywords.letter
        if has_keyword_flag(path.name, papaya_letter):
            # User correction or confirmation - strip the flag
            new_name = remove_keyword_flag(path.name, papaya_letter)
            if new_name != path.name:
                new_path = path.parent / new_name
                path.rename(new_path)
                path = new_path
                LOGGER.info("Stripped papaya flag from %s (user override)", message_id)

        # Train on this mail
        message = read_message(path)
        self.rule_engine.execute_train(self.name, message, category)
        LOGGER.info("Trained on %s → %s", message_id, category)
```

### 4.2 Training Flow Diagram

```
Mail appears in category folder (watcher event)
                │
                ▼
┌───────────────────────────────┐
│ Is message_id in auto_cache?  │
│ (TTL ~5 minutes)              │
└───────────────────────────────┘
        │               │
       YES              NO
        │               │
        ▼               ▼
┌─────────────┐  ┌───────────────────────────┐
│ Skip train  │  │ Does filename have        │
│ (daemon     │  │ papaya flag letter?       │
│  just moved)│  └───────────────────────────┘
└─────────────┘          │               │
                        YES              NO
                         │               │
                         ▼               │
              ┌─────────────────────┐    │
              │ Strip papaya flag   │    │
              │ (atomic rename)     │    │
              │ User is correcting  │    │
              └─────────────────────┘    │
                         │               │
                         └───────┬───────┘
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │ Execute train_rules   │
                    │ (user sorted this)    │
                    └───────────────────────┘
```

---

## 5. Configuration Schema Changes

### 5.1 Updated Config Schema

```yaml
# ~/.config/papaya/config.yaml

# Root directory for models, logs, etc.
root_dir: ~/.local/lib/papaya

# Module search paths (optional - builtin modules always available)
module_paths:
  - ~/.config/papaya/modules    # User overrides
  - /opt/papaya/custom          # Shared custom modules

# Global classification rules (required)
rules: |
  features = modules.extract_features.classify(message)
  bayes = modules.naive_bayes.classify(message, features, account)

  if features.is_malformed:
      move_to("Spam", confidence=0.99)

  if bayes.scores.get("Spam", 0) > 0.85:
      move_to("Spam", confidence=bayes.scores["Spam"])

  if features.has_list_unsubscribe and bayes.scores.get("Spam", 0) < 0.5:
      move_to("Newsletters", confidence=0.8)

  if features.domain_mismatch_score > 3:
      move_to("Spam", confidence=0.9)

# Global training rules (required)
train_rules: |
  features = modules.extract_features.classify(message)
  modules.naive_bayes.train(message, features, category, account)
  modules.tfidf_sgd.train(message, features, category, account)

# Mail accounts
maildirs:
  - name: personal
    path: /var/vmail/me
    # Optional: account-specific rules override global
    rules: |
      features = modules.extract_features.classify(message)
      bayes = modules.naive_bayes.classify(message, features, account)

      # Personal account: more aggressive spam filtering
      if bayes.scores.get("Spam", 0) > 0.7:
          move_to("Spam")
      else:
          fallback()  # Use global rules
    # train_rules: omitted = use global train_rules

  - name: work
    path: /var/vmail/work
    rules: |
      features = modules.extract_features.classify(message)

      # Work account: whitelist company domain
      if "@mycompany.com" in features.from_address.lower():
          skip()  # Always inbox
      else:
          fallback()

# Category definitions
categories:
  Spam:
    flag: spam      # Blacklist senders
  Newsletters:
    flag: neutral   # ML only
  Important:
    flag: ham       # Whitelist senders

# Logging configuration
logging:
  level: info
  debug_file: false
```

### 5.2 Config Dataclass Updates

```python
# src/papaya/config.py (additions)

@dataclass(frozen=True)
class MaildirAccountConfig:
    """Extended maildir account configuration."""
    name: str
    path: Path
    rules: str | None = None        # Account-specific classification rules
    train_rules: str | None = None  # Account-specific training rules


@dataclass(frozen=True)
class Config:
    """Application configuration."""
    root_dir: Path
    module_paths: list[Path]        # NEW
    rules: str                      # NEW: global classification rules
    train_rules: str                # NEW: global training rules
    maildirs: list[MaildirAccountConfig]
    categories: dict[str, CategoryConfig]
    logging: LoggingConfig
```

---

## 6. Implementation Order

### Phase 1: Dovecot Integration (Low Risk, Isolated)

**Files:**
- `src/papaya/dovecot.py` (new)
- `src/papaya/maildir.py` (add flag functions)

**Tests:**
- `tests/unit/test_dovecot.py`
- `tests/unit/test_maildir_flags.py`

**Deliverable:** Can parse/write dovecot-keywords, manipulate filename flags.

---

### Phase 2: Mover Flag Support

**Files:**
- `src/papaya/mover.py` (add papaya_flag parameter)

**Tests:**
- `tests/unit/test_mover.py` (update existing)

**Deliverable:** Mover can add papaya flag when moving mail.

---

### Phase 3: Training Guard

**Files:**
- `src/papaya/runtime.py` (update _handle_user_sort)

**Tests:**
- `tests/integration/test_training_guard.py`

**Deliverable:** Daemon-sorted mail doesn't trigger training; user corrections do.

---

### Phase 4: Module System

**Files:**
- `src/papaya/modules/__init__.py`
- `src/papaya/modules/loader.py`
- `src/papaya/modules/context.py`
- `src/papaya/modules/extract_features.py`
- `src/papaya/modules/naive_bayes.py`
- `src/papaya/modules/tfidf_sgd.py`

**Tests:**
- `tests/unit/test_module_loader.py`
- `tests/unit/test_builtin_modules.py`

**Deliverable:** Modules load, startup/cleanup hooks work.

---

### Phase 5: Rule Engine

**Files:**
- `src/papaya/rules.py` (new)
- `src/papaya/config.py` (add rules fields)

**Tests:**
- `tests/unit/test_rules.py`
- `tests/unit/test_rule_compilation.py`

**Deliverable:** Rules compile, execute, return decisions.

---

### Phase 6: Pipeline Integration

**Files:**
- `src/papaya/runtime.py` (replace pipeline with rule engine)
- `src/papaya/cli.py` (update daemon setup)

**Tests:**
- `tests/integration/test_rule_pipeline.py`
- `tests/e2e/test_daemon_with_rules.py`

**Deliverable:** Daemon uses rule engine for classification.

---

### Phase 7: Cleanup & Migration

**Files to deprecate:**
- `src/papaya/pipeline.py` → keep for reference, mark deprecated
- `src/papaya/classifiers/registry.py` → functionality moved to modules

**Documentation:**
- Update README with new config format
- Add migration guide from old config

---

## 7. Testing Strategy

### Unit Tests

```python
# tests/unit/test_dovecot.py

def test_ensure_keyword_creates_file(tmp_path):
    """Keyword file created when missing."""
    maildir = tmp_path / "maildir"
    maildir.mkdir()

    dk = DovecotKeywords(maildir)
    letter = dk.ensure_keyword()

    assert letter == "a"
    assert (maildir / "dovecot-keywords").exists()


def test_ensure_keyword_reuses_existing(tmp_path):
    """Reuses existing keyword slot."""
    maildir = tmp_path / "maildir"
    maildir.mkdir()
    (maildir / "dovecot-keywords").write_text("0 $Junk\n1 $PapayaSorted\n")

    dk = DovecotKeywords(maildir)
    letter = dk.ensure_keyword()

    assert letter == "b"  # Index 1 = 'b'


def test_ensure_keyword_finds_free_slot(tmp_path):
    """Finds first free slot when others taken."""
    maildir = tmp_path / "maildir"
    maildir.mkdir()
    (maildir / "dovecot-keywords").write_text("0 $Junk\n2 $Other\n")

    dk = DovecotKeywords(maildir)
    letter = dk.ensure_keyword()

    assert letter == "b"  # Index 1 is free


# tests/unit/test_maildir_flags.py

def test_parse_maildir_info():
    assert parse_maildir_info("123.host:2,RSab") == ("123.host", "RS", "ab")
    assert parse_maildir_info("123.host:2,") == ("123.host", "", "")
    assert parse_maildir_info("123.host") == ("123.host", "", "")


def test_add_keyword_flag():
    assert add_keyword_flag("123:2,RS", "a") == "123:2,RSa"
    assert add_keyword_flag("123:2,RSa", "a") == "123:2,RSa"  # Idempotent
    assert add_keyword_flag("123:2,RSb", "a") == "123:2,RSab"  # Sorted


def test_remove_keyword_flag():
    assert remove_keyword_flag("123:2,RSab", "a") == "123:2,RSb"
    assert remove_keyword_flag("123:2,RS", "a") == "123:2,RS"  # No-op


# tests/unit/test_rules.py

class MockStore:
    def log_predictions(self, *_args, **_kwargs):
        pass

def test_rule_move_to():
    """move_to() sets decision correctly."""
    loader = MockModuleLoader()
    engine = RuleEngine(loader, MockStore(), "move_to('Spam', confidence=0.9)", "pass")

    decision = engine.execute_classify("test", MockMessage(), message_id="<msg>")

    assert decision.action == "move"
    assert decision.category == "Spam"
    assert decision.confidence == 0.9


def test_rule_fallback():
    """fallback() defers to global rules."""
    loader = MockModuleLoader()
    engine = RuleEngine(
        loader,
        MockStore(),
        global_rules="move_to('Default')",
        global_train_rules="pass",
    )
    engine.set_account_rules("personal", "fallback()", None)

    decision = engine.execute_classify("personal", MockMessage(), message_id="<msg>")

    assert decision.category == "Default"


def test_rule_syntax_error():
    """Syntax errors raise RuleError at compile time."""
    loader = MockModuleLoader()

    with pytest.raises(RuleError) as exc_info:
        RuleEngine(loader, MockStore(), "if True", "pass")  # Missing colon

    assert "Syntax error" in str(exc_info.value)
```

### Integration Tests

```python
# tests/integration/test_training_guard.py

def test_daemon_move_does_not_train(tmp_maildir, rule_engine, trainer):
    """Mail moved by daemon should not trigger training."""
    # Setup
    msg_path = create_test_message(tmp_maildir / "new")
    mover = MailMover(tmp_maildir, papaya_flag="a")
    auto_cache = AutoClassificationCache()

    # Simulate daemon classification
    message_id = extract_message_id(msg_path)
    dest = mover.move_to_category(msg_path, "Spam", add_papaya_flag=True)
    auto_cache.add(message_id)

    # Verify flag was added
    assert has_keyword_flag(dest.name, "a")

    # Simulate watcher event
    if auto_cache.consume(message_id):
        trained = False
    else:
        trainer.on_user_sort(dest, "Spam")
        trained = True

    assert not trained


def test_user_correction_triggers_training(tmp_maildir):
    """User moving mail to different category should train."""
    # Setup: mail in Spam with papaya flag (daemon put it there)
    spam_dir = tmp_maildir / ".Spam" / "cur"
    spam_dir.mkdir(parents=True)
    msg_path = spam_dir / "123.host:2,Sa"  # 'a' = papaya flag
    msg_path.write_bytes(b"From: test@example.com\n\nBody")

    # User moves to Important (correction)
    important_dir = tmp_maildir / ".Important" / "cur"
    important_dir.mkdir(parents=True)
    new_path = important_dir / msg_path.name
    msg_path.rename(new_path)

    # Training handler should strip flag and train
    dk = DovecotKeywords(tmp_maildir)
    dk._letter = "a"

    if has_keyword_flag(new_path.name, "a"):
        stripped_name = remove_keyword_flag(new_path.name, "a")
        final_path = new_path.parent / stripped_name
        new_path.rename(final_path)
        assert not has_keyword_flag(final_path.name, "a")
```

### E2E Tests

```python
# tests/e2e/test_daemon_with_rules.py

def test_full_classification_flow(tmp_maildir, tmp_config):
    """End-to-end test of rule-based classification."""
    config = f"""
module_paths: []
rules: |
  features = modules.extract_features.classify(message)
  if "viagra" in features.body_text.lower():
      move_to("Spam")
train_rules: |
  pass
maildirs:
  - name: test
    path: {tmp_maildir}
categories:
  Spam:
    flag: spam
logging:
  level: debug
"""
    # ... setup and run daemon ...

    # Deliver spam message
    deliver_message(tmp_maildir / "new", body="Buy Viagra now!")
    time.sleep(1)

    # Verify moved to Spam with flag
    spam_messages = list((tmp_maildir / ".Spam" / "cur").iterdir())
    assert len(spam_messages) == 1
    assert has_keyword_flag(spam_messages[0].name, "a")  # papaya flag
```

---

## 8. Migration Guide

### From Old Config to New Config

**Old format:**
```yaml
maildirs:
  - name: personal
    path: /var/vmail/me

categories:
  Spam:
    min_confidence: 0.85
    flag: spam

classifiers:
  - name: naive_bayes
    type: naive_bayes
    mode: active
```

**New format:**
```yaml
maildirs:
  - name: personal
    path: /var/vmail/me

rules: |
  features = modules.extract_features.classify(message)
  bayes = modules.naive_bayes.classify(message, features, account)
  if bayes.scores.get("Spam", 0) > 0.85:
      move_to("Spam")

train_rules: |
  features = modules.extract_features.classify(message)
  modules.naive_bayes.train(message, features, category, account)

categories:
  Spam:
    flag: spam
```

### Key Changes

1. **`classifiers` section removed** — Modules are discovered automatically
2. **`min_confidence` moved to rules** — Thresholds are now in rule logic
3. **`rules` and `train_rules` required** — Must define classification/training logic
4. **Account `rules` optional** — Falls back to global if not specified

### Backward Compatibility

Legacy classifier-centric configs are no longer accepted. Administrators must migrate to the rule-based schema before upgrading; the loader now requires `rules`/`train_rules` to be present and fails fast otherwise.
