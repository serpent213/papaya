# Papaya Implementation Plan

## Overview

This document outlines the implementation strategy for Papaya, a maildir-monitoring spam classification daemon. The plan is structured in phases with explicit dependencies, milestones, and a comprehensive test suite architecture.

---

## Project Setup

### Initial Structure

```
papaya/
├── pyproject.toml          # uv-managed dependencies
├── src/
│   └── papaya/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── daemon.py
│       ├── watcher.py
│       ├── extractor/
│       │   ├── __init__.py
│       │   ├── features.py
│       │   ├── html.py
│       │   └── domain.py
│       ├── classifiers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── naive_bayes.py
│       │   └── tfidf_sgd.py
│       ├── trainer.py
│       ├── mover.py
│       ├── senders.py
│       ├── store.py
│       └── logging.py
├── tests/
│   ├── conftest.py         # Shared fixtures
│   ├── fixtures/           # Test email samples
│   │   ├── spam/
│   │   ├── ham/
│   │   └── malformed/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── docs/
```

### Dependencies

```toml
[project]
dependencies = [
    "watchdog>=4.0",
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "scikit-learn>=1.4",
    "pyyaml>=6.0",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-asyncio>=0.23",
    "pytest-timeout>=2.3",
    "hypothesis>=6.100",
    "freezegun>=1.4",
    "faker>=24.0",
]
```

---

## Phase 1: Foundation & Core Types

**Goal**: Establish project skeleton, core data types, and configuration system.

### 1.1 Project Initialisation

- [ ] Initialise project with `uv init`
- [ ] Configure pyproject.toml with dependencies
- [ ] Set up src layout with `papaya` package
- [ ] Configure pytest and coverage

### 1.2 Core Data Types (`types.py`)

Define immutable data structures used throughout:

```python
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Protocol

class Category(str, Enum):
    """Mail categories - extensible via config"""
    SPAM = "Spam"
    NEWSLETTERS = "Newsletters"
    IMPORTANT = "Important"

class FolderFlag(Enum):
    """Classification behaviour flags"""
    HAM = auto()      # Whitelist sender
    SPAM = auto()     # Blacklist sender
    NEUTRAL = auto()  # ML only

class ClassifierMode(Enum):
    """Classifier operational mode"""
    ACTIVE = auto()   # Used for sorting
    SHADOW = auto()   # Logging only

@dataclass(frozen=True)
class Features:
    """Extracted email features for classification"""
    body_text: str
    subject: str
    from_address: str
    from_display_name: str
    has_list_unsubscribe: bool
    x_mailer: str | None
    link_count: int
    image_count: int
    has_form: bool
    domain_mismatch_score: float
    is_malformed: bool

@dataclass(frozen=True)
class Prediction:
    """Classification result"""
    category: Category | None  # None = inbox
    confidence: float
    scores: dict[Category, float]

@dataclass(frozen=True)
class MaildirAccount:
    """Configured maildir account"""
    name: str
    path: Path

@dataclass(frozen=True)
class CategoryConfig:
    """Per-category configuration"""
    name: str
    min_confidence: float
    flag: FolderFlag
```

### 1.3 Configuration System (`config.py`)

- [ ] YAML schema definition
- [ ] Config loading with validation
- [ ] Default values and path expansion
- [ ] Environment variable overrides

```python
@dataclass
class Config:
    root_dir: Path
    maildirs: list[MaildirAccount]
    categories: dict[str, CategoryConfig]
    classifiers: list[ClassifierConfig]
    logging: LoggingConfig

def load_config(path: Path | None = None) -> Config:
    """Load and validate configuration from YAML"""
    ...
```

### 1.4 Logging Setup (`logging.py`)

- [ ] Structured logging configuration
- [ ] Main log (INFO) + debug log (DEBUG) separation
- [ ] Log rotation setup
- [ ] Coloured console output (symbols only)

### Tests for Phase 1

**Unit Tests**:
- `test_types.py`: Data class instantiation, immutability, enum membership
- `test_config.py`: Valid/invalid YAML parsing, defaults, path expansion, validation errors

**Integration Tests**:
- Config file loading from disk
- Logging to actual files

---

## Phase 2: Email Parsing & Feature Extraction

**Goal**: Parse emails and extract all classification features.

### 2.1 MIME Parsing (`extractor/features.py`)

- [ ] Parse multipart MIME messages
- [ ] Extract plain text body (prefer text/plain over text/html)
- [ ] Charset detection and decoding with fallbacks
- [ ] Handle malformed emails gracefully (mark as malformed)

### 2.2 Header Extraction

- [ ] From address + display name parsing
- [ ] Subject extraction and decoding
- [ ] List-Unsubscribe detection
- [ ] X-Mailer / User-Agent extraction
- [ ] Received chain analysis

### 2.3 HTML Analysis (`extractor/html.py`)

- [ ] HTML to text conversion (for body)
- [ ] Link extraction and counting
- [ ] Image counting
- [ ] Form element detection

### 2.4 Domain Mismatch Scoring (`extractor/domain.py`)

- [ ] Extract domain from From address
- [ ] Extract domains from all links
- [ ] Calculate mismatch score (links with different 1st/2nd level domain)
- [ ] Handle edge cases (IP addresses, localhost, etc.)

### 2.5 Feature Assembly

- [ ] Combine all extractors into unified `Features` object
- [ ] Handle extraction failures gracefully
- [ ] Normalise text (lowercase, strip excessive whitespace)

### Tests for Phase 2

**Unit Tests**:
- `test_features.py`: Feature extraction from various email formats
- `test_html.py`: HTML parsing, link/image counting, form detection
- `test_domain.py`: Domain extraction, mismatch scoring edge cases
- `test_mime.py`: MIME parsing, multipart handling, charset fallbacks

**Property-Based Tests** (Hypothesis):
- Random email generation with known features
- Malformed input fuzzing
- Unicode stress testing

**Test Fixtures**:
- Real spam samples (sanitised)
- Newsletter samples
- Normal correspondence
- Malformed emails (truncated, invalid encoding, nested MIME)
- Phishing attempts (high domain mismatch)

---

## Phase 3: Classifier Infrastructure

**Goal**: Implement pluggable classifier architecture with initial implementations.

### 3.1 Classifier Protocol (`classifiers/base.py`)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Classifier(Protocol):
    name: str

    def train(self, features: Features, label: Category) -> None:
        """Online training - single sample update"""
        ...

    def predict(self, features: Features) -> Prediction:
        """Returns category + confidence scores"""
        ...

    def save(self, path: Path) -> None:
        """Persist model state"""
        ...

    def load(self, path: Path) -> None:
        """Restore model state"""
        ...

    def is_trained(self) -> bool:
        """Check if model has been trained"""
        ...
```

### 3.2 Feature Vectorisation

- [ ] Text vectoriser (CountVectorizer for NB, TfidfVectorizer for SGD)
- [ ] Categorical feature encoding
- [ ] Numeric feature scaling
- [ ] Incremental vocabulary updates

### 3.3 Naive Bayes Classifier (`classifiers/naive_bayes.py`)

- [ ] MultinomialNB with `partial_fit` for online learning
- [ ] Vocabulary management (incremental updates)
- [ ] Prediction with confidence scores
- [ ] Model serialisation (pickle)

### 3.4 TF-IDF + SGD Classifier (`classifiers/tfidf_sgd.py`)

- [ ] TfidfVectorizer with incremental vocabulary
- [ ] SGDClassifier with `partial_fit`
- [ ] Hinge loss or log loss (configurable)
- [ ] Model serialisation

### 3.5 Classifier Registry (`classifiers/registry.py`)

- [ ] Register classifiers by name and type
- [ ] Track active vs shadow mode
- [ ] Broadcast training events to all classifiers
- [ ] Retrieve active classifier for predictions

```python
class ClassifierRegistry:
    def register(self, classifier: Classifier, mode: ClassifierMode) -> None: ...
    def get_active(self) -> Classifier: ...
    def get_all(self) -> list[tuple[Classifier, ClassifierMode]]: ...
    def train_all(self, features: Features, label: Category) -> None: ...
    def predict_all(self, features: Features) -> dict[str, Prediction]: ...
```

### Tests for Phase 3

**Unit Tests**:
- `test_naive_bayes.py`: Training, prediction, serialisation
- `test_tfidf_sgd.py`: Training, prediction, serialisation
- `test_registry.py`: Registration, mode management, broadcast

**Integration Tests**:
- Train classifier, save, load, verify predictions unchanged
- Compare classifier outputs on same input
- Memory usage during training (no leaks)

**Benchmark Tests**:
- Classification latency (target: <50ms per email)
- Training latency (target: <100ms per sample)
- Model size growth over training

---

## Phase 4: Storage & State Management

**Goal**: Implement persistent storage for models, sender lists, and tracking.

### 4.1 Directory Structure (`store.py`)

```
~/.local/lib/papaya/
├── models/
│   ├── {account}/
│   │   ├── naive_bayes.pkl
│   │   └── tfidf_sgd.pkl
│   └── global/
│       ├── naive_bayes.pkl
│       └── tfidf_sgd.pkl
├── {account}/
│   ├── whitelist.txt
│   └── blacklist.txt
├── trained_ids.txt
├── predictions.log
└── logs/
    ├── papaya.log
    └── debug.log
```

### 4.2 Model Persistence

- [ ] Atomic writes (write to temp, rename)
- [ ] Load on startup, save after training
- [ ] Handle missing/corrupted model files

### 4.3 Sender Lists (`senders.py`)

- [ ] Load/save whitelist and blacklist per account
- [ ] Add/remove addresses
- [ ] Rebuild from folder contents on startup
- [ ] Case-insensitive matching
- [ ] Domain extraction for partial matching (future)

### 4.4 Training Deduplication

- [ ] Track trained Message-IDs
- [ ] Persist to disk (append-only log)
- [ ] Bloom filter for fast lookups (optional optimisation)

### 4.5 Prediction Logging

- [ ] Shadow classifier predictions
- [ ] Structured log format for analysis
- [ ] Rotation and cleanup

### Tests for Phase 4

**Unit Tests**:
- `test_store.py`: Path construction, atomic writes, corruption handling
- `test_senders.py`: List operations, case sensitivity, persistence

**Integration Tests**:
- Store survival across restarts
- Concurrent access (multiple daemon instances - should fail gracefully)
- Disk full handling

---

## Phase 5: Maildir Operations

**Goal**: Implement maildir reading, watching, and mail moving.

### 5.1 Maildir Conventions

Understand maildir structure:
```
{maildir}/
├── new/          # Newly delivered, unread
├── cur/          # Read messages
├── tmp/          # Temporary during delivery
├── .Spam/
│   ├── new/
│   ├── cur/
│   └── tmp/
├── .Newsletters/
│   └── ...
└── .Important/
    └── ...
```

### 5.2 Mail Reading

- [ ] List messages in new/ and cur/
- [ ] Parse message from file path
- [ ] Extract Message-ID for tracking

### 5.3 Filesystem Watcher (`watcher.py`)

- [ ] Watch `{maildir}/new/` for incoming mail
- [ ] Watch `.{Category}/new/` and `.{Category}/cur/` for training triggers
- [ ] Debounce rapid changes (email clients may touch files multiple times)
- [ ] Handle watched directory creation/deletion

```python
class MaildirWatcher:
    def __init__(self, maildir: Path, categories: list[str]):
        ...

    def on_new_mail(self, callback: Callable[[Path], None]) -> None:
        """Register callback for new mail in inbox"""
        ...

    def on_user_sort(self, callback: Callable[[Path, Category], None]) -> None:
        """Register callback for mail sorted by user"""
        ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
```

### 5.4 Mail Mover (`mover.py`)

- [ ] Atomic move via maildir conventions
- [ ] Update maildir flags appropriately
- [ ] Generate unique filename in destination
- [ ] Audit logging of all moves

```python
class MailMover:
    def move_to_category(self, msg_path: Path, category: Category) -> Path:
        """Move message to category folder, return new path"""
        ...

    def move_to_inbox(self, msg_path: Path) -> Path:
        """Move message to inbox cur/, return new path"""
        ...
```

### Tests for Phase 5

**Unit Tests**:
- `test_maildir.py`: Path construction, flag parsing
- `test_mover.py`: Atomic moves, filename generation, flag handling

**Integration Tests**:
- Create temp maildir structure, move files, verify structure
- Concurrent moves (no race conditions)
- Watcher event delivery

**E2E Tests**:
- Full maildir with real messages
- Simulate mail delivery (copy to new/)
- Verify classification and move

---

## Phase 6: Core Pipeline Integration

**Goal**: Wire together extraction, classification, and moving into the main pipeline.

### 6.1 Classification Pipeline

```python
class Pipeline:
    def __init__(
        self,
        extractor: FeatureExtractor,
        registry: ClassifierRegistry,
        senders: SenderLists,
        mover: MailMover,
        config: Config,
    ):
        ...

    def process_new_mail(self, msg_path: Path) -> None:
        """Full classification pipeline for new mail"""
        ...
```

Pipeline steps:
1. Extract From address
2. Check blacklist → move to spam folder
3. Check whitelist → keep in inbox (move to cur/)
4. Extract features (malformed → treat as spam)
5. Run all classifiers (active + shadow)
6. Log all predictions
7. If active classifier confidence > threshold → move to category
8. Otherwise → move to inbox cur/

### 6.2 Error Handling

- [ ] Extraction failures: Log and skip (don't crash daemon)
- [ ] Classification failures: Log and treat as unknown (inbox)
- [ ] Move failures: Retry with backoff, alert on persistent failure

### 6.3 Pipeline Metrics

- [ ] Messages processed count
- [ ] Classification breakdown by category
- [ ] Average confidence scores
- [ ] Processing latency histogram

### Tests for Phase 6

**Unit Tests**:
- `test_pipeline.py`: Mock components, verify flow

**Integration Tests**:
- Real email through pipeline
- Blacklist/whitelist bypass
- Malformed email handling

**E2E Tests**:
- Full pipeline with real classifiers
- Multiple accounts
- Error injection (corrupt files, permission errors)

---

## Phase 7: Event-Driven Training

**Goal**: Implement training triggered by user sorting mail.

### 7.1 Training Trigger

When watcher detects mail in category folder:
1. Check if Message-ID already trained → skip
2. Determine label from folder location
3. Update sender lists if folder has ham/spam flag
4. Extract features
5. Train all classifiers (per-account + global)
6. Mark Message-ID as trained

### 7.2 Misclassification Correction

When mail moves between category folders:
1. Detect as removal from old folder + addition to new
2. Determine new label from new folder
3. Retrain with corrected label
4. Update sender lists (remove from old, add to new if applicable)

### 7.3 Trainer Module (`trainer.py`)

```python
class Trainer:
    def __init__(
        self,
        registry: ClassifierRegistry,
        senders: SenderLists,
        store: Store,
        config: Config,
    ):
        ...

    def on_user_sort(self, msg_path: Path, category: Category) -> None:
        """Handle user sorting mail into category"""
        ...

    def initial_training(self, maildir: Path) -> None:
        """Train on existing categorised mail (startup)"""
        ...
```

### 7.4 Global Training

- [ ] Aggregate training data from all accounts
- [ ] Train global models in addition to per-account
- [ ] Global models for monitoring only (not used for classification)

### Tests for Phase 7

**Unit Tests**:
- `test_trainer.py`: Training flow, deduplication, list updates

**Integration Tests**:
- Train on existing maildir
- Simulate user sorting
- Correction handling

**E2E Tests**:
- Cold start training
- Ongoing training during operation
- Model accuracy improvement verification

---

## Phase 8: CLI Implementation

**Goal**: Implement command-line interface for daemon control and utilities.

### 8.1 CLI Structure (`cli.py`)

```
papaya daemon          # Run daemon (foreground)
papaya daemon -d       # Daemonise
papaya status          # Show status
papaya classify <file> # Classify single email
papaya train           # Manual training trigger
papaya train --full    # Full retrain
papaya compare         # Classifier comparison
```

### 8.2 Daemon Command

- [ ] Foreground mode with graceful shutdown (SIGINT, SIGTERM)
- [ ] Background mode with PID file
- [ ] Status checking (is daemon running?)
- [ ] Log streaming option

### 8.3 Status Command

Output:
```
→ Papaya Status

Daemon: ● Running (PID 12345)
Uptime: 3d 14h 22m

Accounts:
  self     /var/vmail/reactor.de/self     1,234 processed
  siggi    /var/vmail/reactor.de/siggi      567 processed

Classifiers:
  naive_bayes [A]  trained: 5,678 samples  accuracy: 94.2%
  tfidf_sgd   [S]  trained: 5,678 samples  accuracy: 96.1%

Recent Activity:
  Last classification: 2m ago
  Last training: 15m ago
  Today: 45 spam, 12 newsletters, 89 inbox
```

### 8.4 Classify Command

- [ ] Load models
- [ ] Extract features from .eml file
- [ ] Run all classifiers
- [ ] Display predictions with confidence

### 8.5 Train Command

- [ ] Trigger training on existing categorised mail
- [ ] `--full` flag: Clear models, retrain from scratch
- [ ] Progress indicator

### 8.6 Compare Command

- [ ] Analyse shadow prediction logs
- [ ] Compare against actual user sorting
- [ ] Display accuracy metrics per classifier

### Tests for Phase 8

**Unit Tests**:
- `test_cli.py`: Command parsing, argument validation

**Integration Tests**:
- Each command with real filesystem
- Error handling (missing config, invalid paths)

**E2E Tests**:
- Full daemon lifecycle
- CLI interaction with running daemon

---

## Phase 9: Daemon & Process Management

**Goal**: Robust daemon implementation with proper lifecycle management.

### 9.1 Daemon Lifecycle (`daemon.py`)

- [ ] Startup sequence (load config, models, start watchers)
- [ ] Main event loop
- [ ] Graceful shutdown (finish current work, save state)
- [ ] Crash recovery (resume from last known state)

### 9.2 Signal Handling

- [ ] SIGTERM: Graceful shutdown
- [ ] SIGINT: Graceful shutdown
- [ ] SIGHUP: Reload configuration
- [ ] SIGUSR1: Dump status to log

### 9.3 PID File Management

- [ ] Write PID on startup
- [ ] Check for stale PID files
- [ ] Cleanup on shutdown

### 9.4 Health Monitoring

- [ ] Watchdog timer (detect hung daemon)
- [ ] Resource monitoring (memory, open files)
- [ ] Error rate tracking

### Tests for Phase 9

**Unit Tests**:
- `test_daemon.py`: Lifecycle states, signal handling

**Integration Tests**:
- Start/stop daemon
- Signal delivery and handling
- PID file management

**E2E Tests**:
- Long-running daemon stability
- Recovery after crash
- Config reload

---

## Phase 10: Polish & Hardening

**Goal**: Production readiness, performance optimisation, edge case handling.

### 10.1 Performance Optimisation

- [ ] Profile classification pipeline
- [ ] Optimise feature extraction (lazy evaluation, caching)
- [ ] Batch processing for initial training
- [ ] Memory-efficient model storage

### 10.2 Edge Cases

- [ ] Very large emails (> 10MB)
- [ ] Deeply nested MIME
- [ ] Unusual charsets
- [ ] Symlinked maildirs
- [ ] Network filesystems (NFS, CIFS)

### 10.3 Security Hardening

- [ ] Input validation (path traversal, injection)
- [ ] Safe pickle loading (restrict classes)
- [ ] Privilege dropping (if run as root)
- [ ] File permission enforcement

### 10.4 Documentation

- [ ] README with quick start
- [ ] Configuration reference
- [ ] Troubleshooting guide
- [ ] Architecture documentation

### Tests for Phase 10

**Stress Tests**:
- High volume mail processing
- Large maildir (100k+ messages)
- Concurrent access patterns

**Security Tests**:
- Path traversal attempts
- Malicious pickle files
- Resource exhaustion

---

## Test Suite Architecture

### Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── fixtures/
│   ├── emails/
│   │   ├── spam/            # Real spam samples
│   │   ├── ham/             # Real legitimate mail
│   │   ├── newsletters/     # Newsletter samples
│   │   └── malformed/       # Edge cases
│   ├── maildirs/            # Test maildir structures
│   └── configs/             # Test configuration files
├── unit/
│   ├── test_types.py
│   ├── test_config.py
│   ├── test_features.py
│   ├── test_html.py
│   ├── test_domain.py
│   ├── test_naive_bayes.py
│   ├── test_tfidf_sgd.py
│   ├── test_registry.py
│   ├── test_store.py
│   ├── test_senders.py
│   ├── test_maildir.py
│   ├── test_mover.py
│   ├── test_pipeline.py
│   ├── test_trainer.py
│   ├── test_cli.py
│   └── test_daemon.py
├── integration/
│   ├── test_config_loading.py
│   ├── test_classifier_persistence.py
│   ├── test_maildir_operations.py
│   ├── test_pipeline_integration.py
│   ├── test_training_flow.py
│   └── test_cli_commands.py
├── e2e/
│   ├── test_full_pipeline.py
│   ├── test_daemon_lifecycle.py
│   ├── test_multi_account.py
│   └── test_cold_start.py
├── property/
│   ├── test_features_properties.py
│   └── test_classifier_properties.py
├── performance/
│   ├── test_classification_latency.py
│   ├── test_training_throughput.py
│   └── test_memory_usage.py
└── security/
    ├── test_input_validation.py
    └── test_pickle_safety.py
```

### Key Fixtures (`conftest.py`)

```python
import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_maildir(tmp_path: Path) -> Path:
    """Create temporary maildir structure"""
    maildir = tmp_path / "maildir"
    for subdir in ["new", "cur", "tmp"]:
        (maildir / subdir).mkdir(parents=True)
    for category in ["Spam", "Newsletters", "Important"]:
        for subdir in ["new", "cur", "tmp"]:
            (maildir / f".{category}" / subdir).mkdir(parents=True)
    return maildir

@pytest.fixture
def sample_email() -> bytes:
    """Generate sample email content"""
    return b"""From: sender@example.com
To: recipient@example.org
Subject: Test Email
Date: Mon, 1 Jan 2024 12:00:00 +0000
Message-ID: <test-123@example.com>
Content-Type: text/plain; charset=utf-8

This is a test email body.
"""

@pytest.fixture
def spam_email() -> bytes:
    """Generate spam-like email"""
    ...

@pytest.fixture
def temp_config(tmp_path: Path, temp_maildir: Path) -> Path:
    """Create temporary config file"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
maildirs:
  - name: test
    path: {temp_maildir}
categories:
  Spam:
    min_confidence: 0.85
    flag: spam
  Newsletters:
    min_confidence: 0.70
  Important:
    min_confidence: 0.80
    flag: ham
classifiers:
  - name: naive_bayes
    type: naive_bayes
    mode: active
""")
    return config_path

@pytest.fixture
def trained_classifier(sample_emails: list[tuple[bytes, str]]) -> Classifier:
    """Provide pre-trained classifier for testing"""
    ...
```

### Test Categories

#### Unit Tests
- Fast, isolated, no I/O
- Mock all external dependencies
- One assertion focus per test
- Target: 100% coverage of business logic

#### Integration Tests
- Test component interactions
- Real filesystem operations
- Real but isolated classifiers
- Target: Cover all integration points

#### E2E Tests
- Full system tests
- Real daemon, real maildirs
- Simulate actual usage patterns
- Target: Cover critical user journeys

#### Property-Based Tests (Hypothesis)
- Verify invariants hold for all inputs
- Fuzz testing for parsers
- Stress test edge cases

#### Performance Tests
- Latency benchmarks
- Throughput measurements
- Memory profiling
- Regression detection

#### Security Tests
- Input validation
- Injection prevention
- Safe deserialisation

### Coverage Requirements

| Component | Min Coverage |
|-----------|--------------|
| Core types | 100% |
| Config | 95% |
| Feature extraction | 95% |
| Classifiers | 90% |
| Storage | 90% |
| Maildir operations | 90% |
| CLI | 85% |
| Daemon | 80% |
| **Overall** | **90%** |

### CI Integration

```yaml
# .github/workflows/test.yml
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: astral-sh/setup-uv@v4
    - run: uv sync --all-extras
    - run: uv run pytest tests/unit -v --cov
    - run: uv run pytest tests/integration -v
    - run: uv run pytest tests/e2e -v --timeout=60
    - run: uv run pytest tests/property -v
```

---

## Milestones

### M1: Foundation Complete
- [x] Project setup
- [ ] Core types defined
- [ ] Config system working
- [ ] Unit tests passing

### M2: Feature Extraction Complete
- [ ] Email parsing working
- [ ] All features extracted
- [ ] Test fixtures established
- [ ] Property tests passing

### M3: Classification Working
- [ ] Naive Bayes implemented
- [ ] TF-IDF+SGD implemented
- [ ] Registry working
- [ ] Classifier tests passing

### M4: Storage & Maildir Complete
- [ ] State persistence working
- [ ] Sender lists working
- [ ] Maildir operations working
- [ ] Integration tests passing

### M5: Core Pipeline Complete
- [ ] Full classification pipeline
- [ ] Mail moving working
- [ ] E2E tests passing

### M6: Training Complete
- [ ] Event-driven training
- [ ] Correction handling
- [ ] Global training
- [ ] Training tests passing

### M7: CLI Complete
- [ ] All commands implemented
- [ ] Help text polished
- [ ] CLI tests passing

### M8: Production Ready
- [ ] Daemon mode stable
- [ ] Performance optimised
- [ ] Security hardened
- [ ] Documentation complete
- [ ] All tests passing
- [ ] Coverage targets met

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Classifier accuracy poor | High | Shadow mode comparison, easy switching |
| Performance too slow | Medium | Profile early, lazy evaluation |
| Maildir corruption | High | Atomic operations, verification |
| Model corruption | Medium | Checksums, backup on save |

### Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| False positive spam | High | Conservative thresholds, whitelist |
| Missed spam | Medium | Continuous training, blacklist |
| Daemon crash | Medium | Graceful recovery, no state in memory |

---

## Future Considerations (Out of Scope)

- Web UI for configuration
- Remote management API
- Distributed training
- Deep learning models (BERT, etc.)
- Attachment analysis
- URL reputation checking
- Integration with spam databases (SpamAssassin, etc.)

These are explicitly out of scope for initial implementation but the architecture should not preclude them.
