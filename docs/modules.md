# Papaya Rule Engine Refactor

## Overview

Replace the rigid pipeline with a flexible, user-configurable rule engine. Three major components:

1. **Module System** - Hot-reloadable Python modules with lifecycle hooks
2. **Rule DSL** - Python snippets embedded in YAML orchestrating modules
3. **Dovecot Keyword Integration** - `p` flag marks daemon-sorted mail; training distinguishes user vs daemon moves

---

## 1. Module System

### Directory Structure

```
src/papaya/modules/           # Built-in modules (shipped with package)
├── __init__.py
├── extract_features.py       # Wraps existing extractor
├── naive_bayes.py            # Wraps existing classifier
└── tfidf_sgd.py

~/.config/papaya/modules/     # User modules (optional override path)
└── custom_shiboleth.py
```

### Module Interface

Each module is a Python file or package exposing optional hooks:

```python
# modules/naive_bayes.py

def startup(config: ModuleConfig) -> None:
    """Called on daemon start. Load models, initialise state."""
    ...

def classify(message: EmailMessage, features: Features) -> Any:
    """Called during classification. Return structured data for rules."""
    ...

def train(message: EmailMessage, features: Features, category: str) -> None:
    """Called when user sorts mail. Update model."""
    ...

def cleanup() -> None:
    """Called on daemon shutdown. Persist state, release resources."""
    ...
```

All hooks are optional. Missing hook → `AttributeError` if called from snippet (explicit failure).

### Module Loader (`src/papaya/modules/loader.py`)

```python
class ModuleLoader:
    def __init__(self, paths: list[Path]):
        """paths = [builtin_modules_dir, *user_module_dirs]"""
        self._paths = paths
        self._modules: dict[str, ModuleType] = {}

    def load_all(self) -> None:
        """Discover and import all modules from paths."""
        # Later paths override earlier (user overrides builtin)

    def reload_all(self) -> None:
        """Hot-reload all modules (SIGHUP handler)."""

    def get(self, name: str) -> ModuleType:
        """Get module by name. Raises KeyError if not found."""

    def call_startup(self, config: Config) -> None:
        """Call startup() on all modules that have it."""

    def call_cleanup(self) -> None:
        """Call cleanup() on all modules that have it."""
```

### Config Schema Addition

```yaml
# Optional: additional module directories (searched after builtins)
module_paths:
  - ~/.config/papaya/modules
  - /opt/papaya/custom-modules
```

---

## 2. Rule DSL Engine

### Config Schema

```yaml
# Global rules (fallback for accounts without custom rules)
rules: |
  features = modules.extract_features.classify(message, None)
  bayes = modules.naive_bayes.classify(message, features)

  if bayes.scores["Spam"] > 0.85:
      move_to("Spam")
  elif features.has_list_unsubscribe:
      move_to("Newsletters")
  # else: stays in inbox

train_rules: |
  features = modules.extract_features.classify(message, None)
  modules.naive_bayes.train(message, features, category)
  modules.tfidf_sgd.train(message, features, category)

# Per-account overrides
maildirs:
  - name: personal
    path: /var/vmail/me
    rules: |
      # More aggressive spam filtering for personal
      features = modules.extract_features.classify(message, None)
      bayes = modules.naive_bayes.classify(message, features)
      if bayes.scores["Spam"] > 0.7:
          move_to("Spam")
      else:
          fallback()  # Run global rules
    # train_rules: omitted = use global train_rules
```

### Rule Engine (`src/papaya/rules.py`)

```python
@dataclass(frozen=True)
class RuleDecision:
    """Result of rule execution."""
    action: str  # "move" | "inbox" | "fallback"
    category: str | None
    confidence: float | None

class RuleEngine:
    def __init__(self, loader: ModuleLoader, global_rules: str, global_train_rules: str):
        self._loader = loader
        self._global_rules = compile(global_rules, "<global_rules>", "exec")
        self._global_train_rules = compile(global_train_rules, "<global_train_rules>", "exec")
        self._account_rules: dict[str, CodeType] = {}
        self._account_train_rules: dict[str, CodeType] = {}

    def set_account_rules(self, account: str, rules: str | None, train_rules: str | None) -> None:
        """Compile and cache account-specific rules."""

    def execute_classify(self, account: str, message: EmailMessage) -> RuleDecision:
        """Run classification rules for account (with global fallback)."""
        namespace = self._build_namespace(message)

        # Try account rules first
        if account in self._account_rules:
            exec(self._account_rules[account], namespace)
            if namespace.get("_decision"):
                return namespace["_decision"]

        # Fallback to global
        exec(self._global_rules, namespace)
        return namespace.get("_decision") or RuleDecision("inbox", None, None)

    def execute_train(self, account: str, message: EmailMessage, category: str) -> None:
        """Run training rules for account."""
        namespace = self._build_namespace(message, category=category)
        rules = self._account_train_rules.get(account) or self._global_train_rules
        exec(rules, namespace)

    def _build_namespace(self, message: EmailMessage, category: str | None = None) -> dict:
        """Build execution namespace with helpers."""
        decision_holder = {}

        def move_to(cat: str, confidence: float = 1.0):
            decision_holder["_decision"] = RuleDecision("move", cat, confidence)

        def skip():
            decision_holder["_decision"] = RuleDecision("inbox", None, None)

        def fallback():
            decision_holder["_decision"] = RuleDecision("fallback", None, None)

        return {
            "message": message,
            "category": category,  # Only set in train context
            "modules": ModuleNamespace(self._loader),
            "move_to": move_to,
            "skip": skip,
            "fallback": fallback,
            "_decision": None,
            # Safety: no __builtins__ override needed since admin-only
        }

class ModuleNamespace:
    """Attribute access wrapper for modules dict."""
    def __init__(self, loader: ModuleLoader):
        self._loader = loader

    def __getattr__(self, name: str):
        return self._loader.get(name)
```

---

## 3. Dovecot Keyword Integration

### Keyword Management (`src/papaya/dovecot.py`)

```python
PAPAYA_KEYWORD = "$PapayaSorted"
KEYWORD_LETTER = None  # Determined at runtime

class DovecotKeywords:
    """Manage dovecot-keywords file for a maildir."""

    def __init__(self, maildir: Path):
        self._path = maildir / "dovecot-keywords"
        self._letter: str | None = None

    def ensure_keyword(self) -> str:
        """Register $PapayaSorted keyword, return assigned letter (a-z).

        Raises if all 26 slots taken or our keyword conflicts.
        """
        existing = self._load()

        # Check if already registered
        for idx, name in existing.items():
            if name == PAPAYA_KEYWORD:
                self._letter = chr(ord('a') + idx)
                return self._letter

        # Find free slot
        for idx in range(26):
            if idx not in existing:
                existing[idx] = PAPAYA_KEYWORD
                self._save(existing)
                self._letter = chr(ord('a') + idx)
                return self._letter

        raise RuntimeError("No free keyword slots in dovecot-keywords")

    @property
    def letter(self) -> str:
        if self._letter is None:
            raise RuntimeError("Keyword not initialised")
        return self._letter

    def _load(self) -> dict[int, str]:
        """Parse dovecot-keywords file."""
        if not self._path.exists():
            return {}
        result = {}
        for line in self._path.read_text().splitlines():
            if " " in line:
                idx_str, name = line.split(" ", 1)
                result[int(idx_str)] = name
        return result

    def _save(self, keywords: dict[int, str]) -> None:
        """Write dovecot-keywords file."""
        lines = [f"{idx} {name}" for idx, name in sorted(keywords.items())]
        self._path.write_text("\n".join(lines) + "\n")
```

### Filename Flag Operations (`src/papaya/maildir.py` updates)

```python
def parse_maildir_flags(filename: str) -> tuple[str, str]:
    """Parse filename into (base, flags). Example: 'xxx:2,RSa' → ('xxx:2,', 'RSa')"""
    if ":2," in filename:
        base, flags = filename.rsplit(":2,", 1)
        return base + ":2,", flags
    return filename, ""

def add_flag(filename: str, letter: str) -> str:
    """Add flag letter to filename, maintaining sorted order."""
    base, flags = parse_maildir_flags(filename)
    if letter in flags:
        return filename
    new_flags = "".join(sorted(set(flags) | {letter}))
    return base + new_flags

def remove_flag(filename: str, letter: str) -> str:
    """Remove flag letter from filename."""
    base, flags = parse_maildir_flags(filename)
    new_flags = flags.replace(letter, "")
    return base + new_flags

def has_flag(filename: str, letter: str) -> bool:
    """Check if filename has the given flag letter."""
    _, flags = parse_maildir_flags(filename)
    return letter in flags
```

### Updated Mover (`src/papaya/mover.py`)

```python
class MailMover:
    def __init__(self, maildir: Path, *, papaya_flag: str | None = None):
        self._maildir = maildir
        self._papaya_flag = papaya_flag  # Letter from DovecotKeywords

    def move_to_category(self, msg_path: Path, category: str, *, add_papaya_flag: bool = True) -> Path:
        """Move message to category folder, optionally adding papaya flag."""
        dest_dir = self._maildir / f".{category}" / "cur"
        new_name = self._generate_dest_name(msg_path.name)

        if add_papaya_flag and self._papaya_flag:
            new_name = add_flag(new_name, self._papaya_flag)

        dest = dest_dir / new_name
        msg_path.rename(dest)
        return dest
```

### Training Guard Logic

```python
class Trainer:
    def __init__(self, ..., papaya_flag: str, just_moved_cache: JustMovedCache):
        self._papaya_flag = papaya_flag
        self._just_moved = just_moved_cache

    def on_file_appeared(self, msg_path: Path, category: str) -> TrainingResult:
        """Handle file appearing in category folder."""
        message_id = extract_message_id(msg_path)

        # Check if daemon JUST moved this (watcher echo)
        if self._just_moved.pop(message_id, category):
            return TrainingResult(status="skipped_daemon_move", ...)

        # Check for papaya flag (we classified it at some point)
        if has_flag(msg_path.name, self._papaya_flag):
            # User correction or confirmation - strip flag
            new_path = self._strip_papaya_flag(msg_path)
            msg_path = new_path

        # Train on this (user sorted it)
        return self._do_training(msg_path, category)

    def _strip_papaya_flag(self, msg_path: Path) -> Path:
        """Remove papaya flag from filename via atomic rename."""
        new_name = remove_flag(msg_path.name, self._papaya_flag)
        if new_name != msg_path.name:
            new_path = msg_path.parent / new_name
            msg_path.rename(new_path)
            return new_path
        return msg_path


class JustMovedCache:
    """Short-lived cache to suppress watcher echo."""

    def __init__(self, ttl_seconds: float = 5.0):
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[str, float]] = {}  # msg_id → (category, timestamp)
        self._lock = threading.Lock()

    def add(self, message_id: str, category: str) -> None:
        with self._lock:
            self._entries[message_id] = (category, time.time())

    def pop(self, message_id: str, category: str) -> bool:
        """Return True if (msg_id, category) was in cache and matched. Removes entry."""
        with self._lock:
            self._cleanup()
            entry = self._entries.get(message_id)
            if entry and entry[0] == category:
                del self._entries[message_id]
                return True
            return False

    def _cleanup(self) -> None:
        """Remove expired entries."""
        now = time.time()
        self._entries = {k: v for k, v in self._entries.items() if now - v[1] < self._ttl}
```

---

## 4. Pipeline Refactor

Replace `Pipeline` class with rule-engine-based orchestration:

```python
class RulePipeline:
    def __init__(
        self,
        account: str,
        maildir: Path,
        rule_engine: RuleEngine,
        mover: MailMover,
        senders: SenderLists,
        just_moved: JustMovedCache,
    ):
        ...

    def process_new_mail(self, msg_path: Path) -> PipelineResult:
        message = read_message(msg_path)
        message_id = extract_message_id(message)
        from_addr = extract_from(message)

        # Sender list shortcuts (still hardcoded - could move to rules later)
        if self._senders.is_blacklisted(self._account, from_addr):
            dest = self._mover.move_to_category(msg_path, "Spam", add_papaya_flag=True)
            self._just_moved.add(message_id, "Spam")
            return PipelineResult(action="blacklist", category="Spam", ...)

        if self._senders.is_whitelisted(self._account, from_addr):
            dest = self._mover.move_to_inbox(msg_path)
            return PipelineResult(action="whitelist", ...)

        # Execute classification rules
        decision = self._rule_engine.execute_classify(self._account, message)

        if decision.action == "move" and decision.category:
            dest = self._mover.move_to_category(msg_path, decision.category, add_papaya_flag=True)
            self._just_moved.add(message_id, decision.category)
            return PipelineResult(action="classified", category=decision.category, ...)

        dest = self._mover.move_to_inbox(msg_path)
        return PipelineResult(action="inbox", ...)
```

---

## 5. Files to Modify/Create

### New Files
- `src/papaya/modules/loader.py` - Module discovery and loading
- `src/papaya/modules/extract_features.py` - Wrap existing extractor as module
- `src/papaya/modules/naive_bayes.py` - Wrap existing classifier as module
- `src/papaya/modules/tfidf_sgd.py` - Wrap existing classifier as module
- `src/papaya/rules.py` - Rule engine with exec-based DSL
- `src/papaya/dovecot.py` - Keyword file management
- `src/papaya/cache.py` - JustMovedCache implementation

### Modified Files
- `src/papaya/config.py` - Add `rules`, `train_rules`, `module_paths` schema
- `src/papaya/maildir.py` - Add flag parsing/manipulation functions
- `src/papaya/mover.py` - Add papaya flag on daemon moves
- `src/papaya/trainer.py` - Check cache + strip flag before training
- `src/papaya/pipeline.py` - Replace with RulePipeline or refactor to use RuleEngine
- `src/papaya/runtime.py` - Wire up new components, init DovecotKeywords
- `src/papaya/watcher.py` - Pass through to new training logic

### Files to Deprecate/Remove
- `src/papaya/classifiers/registry.py` - Replaced by module system
- Direct classifier imports in pipeline - Now via modules

---

## 6. Implementation Order

1. **Dovecot keyword support** - Low risk, isolated
   - `dovecot.py` - keyword file management
   - `maildir.py` - flag parsing functions
   - `mover.py` - add flag on move

2. **JustMovedCache** - Small, testable
   - `cache.py` implementation
   - Wire into mover and trainer

3. **Training guard logic** - Fix the existing bug
   - Update `trainer.py` to check cache + strip flag
   - Tests to verify daemon moves don't trigger training

4. **Module system** - Foundation for rules
   - `modules/loader.py`
   - Wrap existing extractors/classifiers as modules
   - Config schema for `module_paths`

5. **Rule engine** - Core DSL
   - `rules.py` implementation
   - Config schema for `rules` / `train_rules`
   - Tests with sample snippets

6. **Pipeline integration** - Wire it together
   - Refactor `pipeline.py` to use rule engine
   - Update `runtime.py` initialisation
   - E2E tests

---

## 7. Example Config (Final State)

```yaml
module_paths:
  - ~/.config/papaya/modules

maildirs:
  - name: personal
    path: /var/vmail/me
  - name: work
    path: /var/vmail/work
    rules: |
      # Work account: whitelist company domain
      features = modules.extract_features.classify(message, None)
      if "@mycompany.com" in features.from_address:
          skip()  # Always inbox
      else:
          fallback()

rules: |
  features = modules.extract_features.classify(message, None)
  bayes = modules.naive_bayes.classify(message, features)

  # Malformed = spam
  if features.is_malformed:
      move_to("Spam", confidence=0.99)

  # High-confidence spam
  if bayes.scores["Spam"] > 0.85:
      move_to("Spam", confidence=bayes.scores["Spam"])

  # Newsletter detection
  if features.has_list_unsubscribe and bayes.scores["Spam"] < 0.5:
      move_to("Newsletters", confidence=0.8)

  # Phishing signal
  if features.domain_mismatch_score > 3:
      move_to("Spam", confidence=0.9)

train_rules: |
  features = modules.extract_features.classify(message, None)
  modules.naive_bayes.train(message, features, category)
  modules.tfidf_sgd.train(message, features, category)

categories:
  Spam:
    flag: spam
  Newsletters:
    flag: neutral
  Important:
    flag: ham

logging:
  level: info
```
