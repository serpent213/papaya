# Papaya Spam Eater — PRD

## Overview

A Python daemon that monitors maildir folders, classifies incoming email using pluggable ML algorithms, and automatically sorts mail into category folders. Designed for on-device, CPU-only operation with continuous learning.

---

## Goals

1. **Keep inbox clean** — Move spam and newsletters out, leave important mail in inbox
2. **Learn continuously** — Model improves as user sorts mail into folders
3. **Support experimentation** — Run multiple classifiers in parallel, compare performance
4. **Multi-account** — Watch multiple maildirs with per-account and global training

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Daemon                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │  FS Watcher  │───▶│   Pipeline   │───▶│  Classifier Engine   │   │
│  │  (watchdog)  │    │  (extract)   │    │  ┌────────────────┐  │   │
│  └──────────────┘    └──────────────┘    │  │ NaiveBayes [A] │  │   │
│         │                   │            │  │ TF-IDF+SVM [S] │  │   │
│         │                   │            │  │ ...            │  │   │
│         ▼                   ▼            │  └────────────────┘  │   │
│  ┌──────────────┐    ┌──────────────┐    └──────────────────────┘   │
│  │ Sender Lists │    │   Trainer    │              │                │
│  │ (W/B lists)  │    │  (event)     │              ▼                │
│  └──────────────┘    └──────────────┘    ┌──────────────────────┐   │
│                                          │     Mail Mover       │   │
│                                          └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
        [A] = Active (used for sorting)
        [S] = Shadow (logs predictions only)
```

---

## Components

### 1. Filesystem Watcher

- Uses `watchdog` library to monitor:
  - `{maildir}/new/` — Incoming mail → triggers classification
  - `{maildir}/.{Category}/new/` and `.{Category}/cur/` — User-sorted mail → triggers training
- Does NOT watch inbox `cur/` (no action needed there)

### 2. Feature Extractor

Extracts from each email:

**Text features:**
- **Body text** — Plain text content (decode MIME, strip HTML tags)
- **Subject** — Header field
- **From** — Sender address and display name

**Header features:**
- List-Unsubscribe presence
- X-Mailer / User-Agent
- Received chain patterns

**Structural features (from HTML):**
- Link count
- Image count
- Form element presence
- **Domain mismatch score** — Count of links where 1st/2nd level domain differs from From address domain (phishing signal)

**Encoding fallback**: Best-effort charset decoding. If email is unparseable/malformed after trying common encodings → treat as spam with high confidence (legitimate mail is rarely broken).

Output: Normalised feature dict suitable for vectorisation.

### 3. Classifier Engine

**Pluggable architecture:**
```python
class Classifier(Protocol):
    name: str

    def train(self, features: Features, label: Category) -> None:
        """Online training - single sample update"""
        ...

    def predict(self, features: Features) -> Prediction:
        """Returns category + confidence scores"""
        ...

    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

**Initial implementations:**
1. **NaiveBayes** — Multinomial NB with word counts (classic spam filter)
2. **TF-IDF + SGD** — Scikit-learn SGDClassifier with `partial_fit`

**Classifier modes:**
- `active` — Used for live classification and mail sorting
- `shadow` — Predictions logged but not acted upon (for comparison)

**Training scopes:**
- `per-account` — Separate model per maildir (used for classification)
- `global` — All maildirs combined (monitoring/comparison only)

### 4. State Store

Persists to `~/.local/lib/papaya/`:
- `models/{account}/{classifier}.pkl` — Serialised model state per account
- `models/global/{classifier}.pkl` — Global model state
- `{account}/whitelist.txt` — From addresses from ham-flagged folders
- `{account}/blacklist.txt` — From addresses from spam-flagged folders
- `trained_ids.txt` — Message-IDs already used for training (deduplication)
- `predictions.log` — Shadow classifier predictions for analysis

**Note:** No pointer storage needed — we rely on maildir flags. New mail appears in `new/`, once classified it moves to `cur/` with appropriate flags. On startup, only process files in `new/`.

**Whitelist/blacklist rebuilding**: On startup, scan ham/spam-flagged folders and rebuild lists. During runtime, update incrementally as mail arrives in those folders.

### 5. Trainer

**Event-driven training** via filesystem watcher (not polling):
- Watches all category folders for new arrivals (user sorting mail)
- When mail appears in category folder → train immediately (no delay)
- **Scope**: Trains all registered classifiers (active + shadow), both per-account and global
- **Deduplication**: Tracks already-trained message IDs to avoid retraining

**Misclassification correction**: If mail moves OUT of a category folder (user correcting mistake), the new location is the "truth" — retrain with corrected label.

### 6. Mail Mover

- Moves mail from inbox to category folder based on active classifier prediction
- Respects maildir conventions (atomic move via `cur/` directory)
- Logs all moves for audit trail

---

## Categories & Folder Types

Admin-defined in config. Initial set:
- `Spam` — Unsolicited/malicious mail
- `Newsletters` — Subscriptions, marketing, automated sends
- `Important` — High-priority mail (user-sorted)

Mail not matching any category stays in inbox.

**Folder mapping**: Category name maps to maildir subfolder (e.g., `Spam` → `.Spam/`)

### Folder Classification Flags

Each category folder has a classification flag:
- `ham` — Whitelist folder. From addresses are added to whitelist; future mail from these senders bypasses classification entirely (stays in inbox)
- `spam` — Blacklist folder. From addresses are added to blacklist; future mail from these senders is auto-classified as spam (no ML needed)
- `neutral` — Normal ML classification, no address-based shortcuts

**Example mapping:**
| Folder | Flag | Behaviour |
|--------|------|-----------|
| Important | ham | Whitelist sender |
| Spam | spam | Blacklist sender |
| Newsletters | neutral | ML only |

### Sender Lists

Derived automatically from folder contents:
- **Whitelist**: All From addresses in `ham`-flagged folders
- **Blacklist**: All From addresses in `spam`-flagged folders

**Note:** Inbox cannot be flagged as ham or spam — it's always neutral.

On classification:
1. Check blacklist → move directly to spam-flagged folder (skip ML)
2. Check whitelist → move directly to ham-flagged folder (skip ML)
3. Neither → run ML pipeline

---

## Configuration

`~/.config/papaya/config.yaml` (or set via CLI option):

```yaml
# Optionally override default of ~/.local/lib/papaya/
# rootdir: /var/lib/papaya

# Maildir accounts to monitor
maildirs:
  - name: self
    path: /var/vmail/reactor.de/self
  - name: siggi
    path: /var/vmail/reactor.de/siggi

# Categories with per-category confidence thresholds and flags
categories:
  Spam:
    min_confidence: 0.85    # Higher threshold - false positives costly
    flag: spam              # Blacklist senders
  Newsletters:
    min_confidence: 0.70    # Can be more aggressive
    # flag: neutral           # ML only
  Important:
    min_confidence: 0.80
    flag: ham               # Whitelist senders

# Classifier configuration
classifiers:
  - name: naive_bayes
    type: naive_bayes
    mode: active        # active | shadow
  - name: tfidf_sgd
    type: tfidf_sgd
    mode: shadow
```

---

## CLI

```
papaya daemon          # Run the daemon (foreground)
papaya daemon -d       # Daemonise

papaya status          # Show daemon status, watched maildirs, model stats

papaya classify <file> # Classify a single .eml file
                       # Shows predictions from ALL classifiers:
                       #   naive_bayes [A]: Spam (0.92)
                       #   tfidf_sgd   [S]: Spam (0.87)
                       #   ─────────────────────────
                       #   Consensus: Spam

papaya train           # Trigger manual training run
papaya train --full    # Full retrain (not incremental)

papaya compare         # Show accuracy comparison between classifiers
                       # Uses shadow predictions vs actual user sorting
```

---

## Data Flow

### New Mail Arrival
```
1. Watcher detects new file in {maildir}/new/
2. Extract From address
3. Check blacklist → move to spam-flagged folder (skip ML)
4. Check whitelist → move to ham-flagged folder (skip ML)
5. Extract features (text + structural + domain mismatch)
   - If unparseable → treat as spam (high confidence)
6. All classifiers predict (active + shadow)
7. Log predictions (debug log includes full feature vector)
8. Active classifier result → move if confidence > category threshold
9. Mail moves to cur/ (inbox) or .{Category}/cur/ (sorted)
```

### User Sorts Mail (Training Trigger)
```
1. Watcher detects new file in category folder (user moved it)
2. If folder has ham/spam flag → update whitelist/blacklist with From address
3. Extract features
4. Train all classifiers immediately (per-account + global)
5. Mark message ID as trained
6. If mail was MOVED from another category (correction):
   a. Retrain with new label
   b. If old folder had ham/spam flag → remove From from old list, add to new
```

---

## Logging

**Log files** (in `~/.local/lib/papaya/logs/`):
- `papaya.log` — Main application log (INFO level): daemon start/stop, mail processed, moves, training events, errors
- `debug.log` — Optional debug log (DEBUG level): full classifier input features, prediction scores from all classifiers, model updates

**Configuration:**
```yaml
logging:
  level: info           # info | debug
  debug_file: true      # Enable separate debug.log with classifier I/O
```

**Log format:** Timestamp, level, component, message. Structured enough for parsing but human-readable.

---

## Key Libraries

| Purpose | Library |
|---------|---------|
| Filesystem watching | `watchdog` |
| Email parsing | `email` (stdlib) + `mailbox` |
| HTML parsing | `beautifulsoup4` + `lxml` |
| ML (Naive Bayes) | `scikit-learn` (MultinomialNB) |
| ML (TF-IDF + SGD) | `scikit-learn` (TfidfVectorizer, SGDClassifier) |
| Config | `pyyaml` |
| CLI | `typer` |

---

## File Structure

```
papaya/
├── __init__.py
├── __main__.py           # Entry point
├── cli.py                # CLI commands
├── config.py             # Config loading/validation
├── daemon.py             # Main daemon loop
├── watcher.py            # Filesystem watcher
├── extractor/
│   ├── __init__.py
│   ├── features.py       # Feature dict construction
│   ├── html.py           # HTML parsing, link/image extraction
│   └── domain.py         # Domain mismatch scoring
├── classifiers/
│   ├── __init__.py
│   ├── base.py           # Classifier protocol
│   ├── registry.py       # Classifier registry (active/shadow management)
│   ├── naive_bayes.py
│   └── tfidf_sgd.py
├── trainer.py            # Training orchestration
├── mover.py              # Mail moving logic (maildir operations)
├── senders.py            # Whitelist/blacklist management
├── store.py              # Model persistence, trained IDs tracking
└── logging.py            # Logging configuration (main + debug)
```

---

## Implementation Phases

### Phase 1: Core Pipeline
- Config loading (YAML)
- Maildir watcher (inbox + category folders)
- Feature extraction (text + structural + domain mismatch)
- Single classifier (Naive Bayes)
- Whitelist/blacklist management
- Mail moving
- Basic CLI (daemon, classify, status)

### Phase 2: Training & Multi-Classifier
- Event-driven training (watcher on category folders)
- Misclassification correction handling
- Classifier protocol + registry
- Shadow mode
- Add TF-IDF+SGD classifier
- CLI compare command

### Phase 3: Polish
- Global training (all accounts combined)
- Prediction logging + analysis
- Daemon mode (backgrounding, proper signal handling)
- Comprehensive status output with model stats
