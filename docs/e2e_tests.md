# Integration/E2E Tests for Papaya Email Classification

## Overview

Create comprehensive flow tests that exercise the real ML pipeline with actual filesystem watchers, using statistical assertions for validation.

---

## Test Data: Placeholder Email Corpus

Create `tests/fixtures/corpus/` with placeholder emails:

```
tests/fixtures/corpus/
├── spam/
│   ├── 001.eml    # "Buy cheap viagra now!!!"
│   ├── 002.eml    # "You've won $1,000,000"
│   ├── 003.eml    # "Urgent: Nigerian prince needs help"
│   ├── ...        # ~20 spam samples
│   └── 020.eml
└── ham/
    ├── 001.eml    # "Meeting tomorrow at 3pm"
    ├── 002.eml    # "Project update: Q4 deliverables"
    ├── 003.eml    # "Re: Code review feedback"
    ├── ...        # ~20 ham samples
    └── 020.eml
```

Each `.eml` file is a valid RFC822 message with:
- `From:`, `To:`, `Subject:`, `Message-ID:` headers
- Body text with category-appropriate content

**Rationale**: 40 emails (20 spam, 20 ham) provides enough data for meaningful statistical assertions while keeping tests fast.

---

## Test Files Structure

```
tests/integration/
├── conftest.py                    # Shared fixtures
├── test_training_guard.py         # (existing)
├── test_train_classify.py         # Test 1: Train-then-classify
├── test_user_corrections.py       # Test 2: Learning from corrections
└── test_daemon_lifecycle.py       # Test 3: Daemon lifecycle
```

### Infrastructure

```python
# Event synchronisation helpers
class EventCollector:
    """Collects classification/training events with timeout support."""
    def __init__(self):
        self.events = []
        self._condition = threading.Condition()

    def add(self, event):
        with self._condition:
            self.events.append(event)
            self._condition.notify_all()

    def wait_for(self, count: int, timeout: float = 10.0) -> bool:
        """Wait until `count` events collected or timeout."""
        ...

def load_corpus(corpus_dir: Path) -> dict[str, list[Path]]:
    """Load spam/ham emails from fixture directory."""
    return {
        "Spam": list((corpus_dir / "spam").glob("*.eml")),
        "ham": list((corpus_dir / "ham").glob("*.eml")),
    }

def copy_to_maildir(src: Path, dest_dir: Path) -> Path:
    """Copy email to maildir with proper naming."""
    ...
```

---

## Test 1: Train-Then-Classify Pipeline

**Purpose**: Verify that classifiers trained on corpus can accurately classify new mail.

```
Flow:
1. Create maildir structure with Spam category
2. Copy 70% of corpus to category folders (training set)
3. Wire up real ModuleLoader, RuleEngine, AccountRuntime
4. Run initial_training() to train classifiers
5. Start real MaildirWatcher
6. Drop remaining 30% into inbox/new (validation set)
7. Wait for classification events
8. Assert accuracy > 70% (statistical threshold)
```

**Key assertions**:
- At least 70% of spam emails moved to Spam folder
- At least 70% of ham emails kept in inbox (not moved to Spam)
- No crashes or exceptions during classification

**Timeout**: 30 seconds (accounts for filesystem event delays)

---

## Test 2: Learning From User Corrections

**Purpose**: Verify the training feedback loop improves classification.

```
Flow:
1. Create maildir with empty classifiers (no pre-training)
2. Start AccountRuntime with real watcher
3. "User" sorts half the corpus into correct folders
4. Wait for training events to complete
5. Drop new mail into inbox
6. Verify classification accuracy improves vs. random baseline
7. Simulate user correction (move misclassified mail)
8. Verify correction triggers retraining (check trainer calls)
```

**Key assertions**:
- Training events fire for user-sorted mail
- Classifier accuracy > random (50%) after training
- User correction strips papaya flag and retrains

---

## Test 3: Full Daemon Lifecycle Simulation

**Purpose**: Exercise DaemonRuntime with signal handling and config reload.

```
Flow:
1. Create full config with one account
2. Start DaemonRuntime in background thread
3. Train by sorting corpus into categories
4. Drop validation emails and verify classification
5. Call runtime._handle_sighup() directly - verify modules reload
6. Call runtime.get_status() - verify status snapshot
7. Call runtime.shutdown() - verify clean shutdown
```

**Key assertions**:
- Daemon starts and stops cleanly
- Signal handler methods work correctly (called directly, not via OS signals)
- Classification continues working after reload
- No resource leaks (watcher stopped, files closed)

---

## Implementation Details

### Critical Files to Modify/Create

| File | Action |
|------|--------|
| `tests/fixtures/corpus/spam/*.eml` | Create ~20 spam samples |
| `tests/fixtures/corpus/ham/*.eml` | Create ~20 ham samples |
| `tests/integration/conftest.py` | Shared fixtures (corpus loader, event collector) |
| `tests/integration/test_train_classify.py` | Test 1: Train-then-classify pipeline |
| `tests/integration/test_user_corrections.py` | Test 2: Learning from user corrections |
| `tests/integration/test_daemon_lifecycle.py` | Test 3: Daemon lifecycle simulation |

### Synchronisation Strategy

Since real watchdog events are asynchronous:
1. Use `threading.Condition` to wait for expected event count
2. Poll with generous timeouts (10-30 seconds)
3. Mark tests with `@pytest.mark.timeout(60)` to catch hangs

### Statistical Assertions

```python
def assert_accuracy_above(results: list[tuple[str, str]], threshold: float = 0.7):
    """Assert classification accuracy exceeds threshold."""
    correct = sum(1 for expected, actual in results if expected == actual)
    accuracy = correct / len(results)
    assert accuracy >= threshold, f"Accuracy {accuracy:.1%} below {threshold:.1%}"
```

### Pytest Markers

```python
# In conftest.py or pyproject.toml
pytest.mark.integration  # Skip in unit test runs
pytest.mark.slow         # Optional marker for CI filtering
```

---

## Execution Notes

- Tests use `tmp_path` for isolation — no global state pollution
- Each test gets fresh classifiers (no cross-test contamination)
- Debounce set to 0 for tests to avoid event delays
- Watcher stopped in fixture teardown to prevent resource leaks
