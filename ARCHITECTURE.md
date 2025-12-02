# Papaya Architecture

A Python daemon that monitors maildir folders, classifies incoming email using ML, and continuously learns from user sorting behaviour.

---

## Overview

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

## Directory Structure

```
src/papaya/
├── cli.py                # Typer CLI: daemon, status, classify, train
├── config.py             # YAML config loading and validation
├── dovecot.py            # Dovecot keyword file management
├── logging.py            # Logging configuration
├── maildir.py            # Maildir operations and flag parsing
├── mover.py              # Atomic mail moving with flag support
├── pidfile.py            # PID file management
├── rules.py              # Exec-based rule engine (DSL)
├── runtime.py            # DaemonRuntime + AccountRuntime
├── senders.py            # Whitelist/blacklist management
├── store.py              # Persistence (models, trained IDs, logs)
├── trainer.py            # Event-driven training
├── types.py              # Core data structures
├── watcher.py            # Watchdog filesystem monitoring
│
├── classifiers/          # ML implementations
│   ├── base.py           # Classifier protocol
│   ├── naive_bayes.py    # Multinomial Naive Bayes
│   ├── tfidf_sgd.py      # TF-IDF + SGDClassifier
│   └── vectorizer.py     # Text vectorisation
│
├── extractor/            # Feature extraction
│   ├── features.py       # Main extraction logic
│   ├── html.py           # HTML analysis (links, images, forms)
│   └── domain.py         # Domain mismatch scoring
│
└── modules/              # Hot-reloadable module system
    ├── loader.py         # ModuleLoader
    ├── context.py        # ModuleContext
    ├── extract_features.py
    ├── naive_bayes.py
    └── tfidf_sgd.py
```

---

## Core Components

### Module System

The module system replaces the old classifier registry with a flexible, hot-reloadable plugin architecture.

**ModuleLoader** (`modules/loader.py`)

- Discovers Python modules from configured paths
- Later paths override earlier (user modules override built-ins)
- Manages lifecycle hooks: `startup(ctx)` and `cleanup()`
- Supports hot-reload via SIGHUP

```python
loader = ModuleLoader([builtin_path, user_path])
loader.load_all()
loader.call_startup(ModuleContext(config, store))
module = loader.get("naive_bayes")
loader.reload_all()  # SIGHUP
loader.call_cleanup()
```

**Module Interface** (all hooks optional):

```python
def startup(ctx: ModuleContext) -> None:
    """Load models, initialise state."""

def classify(message: EmailMessage, features: Features, account: str) -> Prediction:
    """Return classification result."""

def train(message: EmailMessage, features: Features, category: str, account: str) -> None:
    """Update model with training sample."""

def cleanup() -> None:
    """Release resources."""
```

**Built-in Modules:**

| Module | Purpose |
|--------|---------|
| `extract_features` | Stateless feature extraction from emails |
| `naive_bayes` | Multinomial Naive Bayes classifier |
| `tfidf_sgd` | TF-IDF vectorisation + SGD classifier |

---

### Rule Engine

The rule engine executes Python snippets embedded in YAML config, replacing the rigid pipeline.

**RuleEngine** (`rules.py`)

Compiles and executes two types of rules:

1. **Classification rules** — run on new mail, decide category
2. **Training rules** — run on user-sorted mail, update models

**Execution namespace provides:**

| Name | Type | Description |
|------|------|-------------|
| `message` | `EmailMessage` | Parsed email |
| `account` | `str` | Account name |
| `category` | `str` | (train only) Destination category |
| `modules` | `ModuleNamespace` | Access to loaded modules |
| `move_to(cat, confidence)` | function | Declare move decision |
| `skip()` | function | Keep in inbox |
| `fallback()` | function | Chain to global rules |

**Example classification rule:**

```python
features = modules.extract_features.classify(message)
bayes = modules.naive_bayes.classify(message, features, account)

if bayes.scores.get("Spam", 0) > 0.85:
    move_to("Spam", confidence=bayes.scores["Spam"])
elif features.has_list_unsubscribe:
    move_to("Newsletters", confidence=0.8)
```

**Rule hierarchy:**
- Account-specific rules execute first
- `fallback()` chains to global rules
- No decision defaults to inbox

---

### Runtime

**DaemonRuntime** (`runtime.py`)

Top-level orchestrator managing multiple accounts:

- Lifecycle management of `AccountRuntime` instances
- Signal handling (SIGTERM, SIGINT, SIGHUP, SIGUSR1)
- Configuration reload callbacks
- Status snapshots

**AccountRuntime** (`runtime.py`)

Bundles all components for a single maildir:

```python
@dataclass
class AccountRuntime:
    name: str
    maildir: Path
    rule_engine: RuleEngine
    senders: SenderLists
    mover: MailMover
    trainer: Trainer
    watcher: MaildirWatcher
    categories: Mapping[str, CategoryConfig]
    papaya_flag: str | None
    auto_cache: AutoClassificationCache
    metrics: ClassificationMetrics
```

---

## Data Flow

### New Mail Classification

```
1. Watcher detects file in {maildir}/new/
2. Read and parse email
3. Check sender shortcuts:
   - Blacklisted → move to Spam (skip ML)
   - Whitelisted → keep in inbox (skip ML)
4. Execute classification rules
5. Apply decision:
   - move_to(category) → mover.move_to_category(add_papaya_flag=True)
   - inbox → mover.move_to_inbox()
6. Record in AutoClassificationCache (5-minute TTL)
7. Update metrics
```

### User Sorting (Training)

```
1. Watcher detects file in category folder
2. Check AutoClassificationCache:
   - If hit → skip (daemon just moved this)
3. Check Papaya flag in filename:
   - If present → strip flag (user correction)
4. Apply sender flag (update whitelist/blacklist)
5. Check duplicate training (trained_ids)
6. Execute training rules
7. Record trained message ID
```

### Training Guard Diagram

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
              │ (user correction)   │    │
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

## Maildir Handling

### Flag Format

Maildir filenames include flags after `:2,`:

```
1701234567890.abc123.hostname:2,RSp
                              │││
                              ││└─ keyword 'p' = $PapayaSorted
                              │└── Seen (standard)
                              └─── Replied (standard)
```

Standard flags are uppercase (D, F, P, R, S, T).
Keyword flags are lowercase (a-z), mapped via `dovecot-keywords`.

### MailMover

Atomic message moving with Papaya flag support:

```python
mover = MailMover(maildir, papaya_flag="p")
mover.move_to_category(path, "Spam", add_papaya_flag=True)
mover.move_to_inbox(path)
```

### DovecotKeywords

Manages the `dovecot-keywords` file:

```
0 $Label1
1 $Label2
15 $PapayaSorted
```

Allocates a letter (a-z) for `$PapayaSorted` on startup.

---

## Persistence

**Store layout:**

```
~/.local/lib/papaya/
├── models/
│   ├── global/
│   │   └── naive_bayes.pkl
│   └── {account}/
│       ├── naive_bayes.pkl
│       └── tfidf_sgd.pkl
├── {account}/
│   ├── whitelist.txt
│   └── blacklist.txt
├── trained_ids.txt
├── predictions.log
└── papaya.pid
```

**Key components:**

| Component | Purpose |
|-----------|---------|
| `Store` | Root manager, classifier save/load |
| `TrainedIdRegistry` | Append-only file tracking trained messages |
| `PredictionLogger` | JSON-lines prediction log with rotation |
| `SenderLists` | Per-account whitelist/blacklist files |

---

## Configuration

```yaml
root_dir: ~/.local/lib/papaya

module_paths:
  - ~/.config/papaya/modules

maildirs:
  - name: personal
    path: /var/vmail/example.com/user
    rules: |           # Optional account override
      # Python snippet
    train_rules: |     # Optional training override
      # Python snippet

rules: |               # Global classification rules
  features = modules.extract_features.classify(message)
  bayes = modules.naive_bayes.classify(message, features, account)
  if bayes.scores.get("Spam", 0) > 0.85:
      move_to("Spam")

train_rules: |         # Global training rules
  features = modules.extract_features.classify(message)
  modules.naive_bayes.train(message, features, category, account)
  modules.tfidf_sgd.train(message, features, category, account)

categories:
  Spam:
    flag: spam         # Blacklist senders
  Newsletters:
    flag: neutral      # ML only
  Important:
    flag: ham          # Whitelist senders

logging:
  level: info
  debug_file: false
```

---

## CLI

```
papaya daemon [-d] [--pid-file PATH] [--skip-initial-training]
papaya status [--pid-file PATH]
papaya classify MESSAGE [-a ACCOUNT]
papaya train [--full] [-a ACCOUNT]
```

---

## Thread Safety

**Thread-safe components:**
- `AutoClassificationCache` — lock around TTL cache
- `TrainedIdRegistry` — lock around file + memory state
- `SenderLists` — lock around cache + file writes
- `PredictionLogger` — lock around rotation + append

**Single-threaded by design:**
- `AccountRuntime` — one watcher thread per account
- `ModuleLoader` — reload only during SIGHUP
- `RuleEngine` — stateless execution

---

## Design Decisions

### Module System Over Registry

Replaced hardcoded classifier registry with dynamic module loading:
- Users can add custom modules without core changes
- Hot-reload support (SIGHUP)
- Module state isolated per account

### Rule DSL Over Pipeline

Replaced fixed pipeline with Python snippets in YAML:
- Users control classifier selection and thresholds
- Account-specific overrides with global fallback
- No code changes needed for new strategies

### Dovecot Keyword for Training Guard

Daemon adds `$PapayaSorted` keyword flag on moves:
- Prevents retraining on daemon's own moves
- User corrections still trigger training (flag removed)
- Survives client sync/cache refresh

### Event-Driven Training

Immediate training on user sort events:
- Continuous learning without manual intervention
- Minimal storage (only trained IDs tracked)
- Fast feedback loop
