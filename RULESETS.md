# Papaya Ruleset Developer Manual

A guide to writing classification and training rules for the Papaya mail sorter.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Configuration](#configuration)
3. [Classification Rules](#classification-rules)
4. [Training Rules](#training-rules)
5. [Built-in Modules](#built-in-modules)
6. [Data Types](#data-types)
7. [Account-Specific Overrides](#account-specific-overrides)
8. [Real-World Examples](#real-world-examples)
9. [Error Handling](#error-handling)
10. [Debugging Tips](#debugging-tips)

---

## Introduction

Papaya rules are Python snippets embedded in your YAML configuration. They execute whenever:

- **Classification rules** — new mail arrives in a maildir's `new/` folder
- **Training rules** — you (or your mail client) move a message into a category folder

Rules have access to the incoming message, the account name, loaded modules, and a set of routing functions. You write ordinary Python—conditionals, loops, string operations—to decide where mail goes.

### Why a Mini-DSL?

A fixed pipeline can't anticipate every sorting strategy. Some users want strict sender whitelists; others prefer ML with high confidence thresholds; others combine both. By exposing Python directly, Papaya lets you express exactly the logic you need without touching core code.

---

## Configuration

Rules live in `~/.config/papaya/config.yaml` (or a path passed via `--config`). They appear as YAML multi-line strings under `rules:` and `train:` keys.

```yaml
rules: |
  # Classification logic here
  features = modules.extract_features.classify(message)
  prediction = modules.naive_bayes.classify(message, features, account)
  if prediction.confidence >= 0.7:
      move_to(prediction.category.value, confidence=prediction.confidence)
  else:
      skip()

train: |
  # Training logic here
  features = modules.extract_features.classify(message)
  modules.naive_bayes.train(message, features, category, account)
```

The `|` character preserves newlines. Indent the Python code consistently (spaces, not tabs).

---

## Classification Rules

Classification rules run once per incoming message. Your job: call one of the routing functions to declare a decision.

### Namespace Reference

| Name | Type | Description |
|------|------|-------------|
| `message` | `EmailMessage` | The parsed email; use `message.get("Header-Name")` to read headers |
| `account` | `str` | The account name from config (e.g., `"personal"`) |
| `message_id` | `str` | The Message-ID header (or a fallback identifier) |
| `modules` | `ModuleNamespace` | Attribute-style access to loaded modules |
| `move_to(category, confidence=1.0)` | function | Route to a category folder |
| `skip()` | function | Deliver to inbox (no routing) |
| `fallback()` | function | Defer to global rules (from account rules only) |
| `log(*args)` / `log_d(*args)` | function | DEBUG-level log helper (alias) |
| `log_i(*args)` | function | INFO-level log helper |
| `log_p(classifier, prediction)` | function | Structured prediction log written to `logs/predictions.log` |

### Routing Functions

**`move_to(category, confidence=1.0)`**

Moves the message to the named category folder. The `category` string must match a key in your `categories:` config block. The optional `confidence` is stored in logs for diagnostics.

```python
move_to("Spam")
move_to("Newsletters", confidence=0.85)
```

**`skip()`**

Delivers the message to the inbox without classification. Use this when you're uncertain or want manual triage.

```python
if prediction.confidence < 0.5:
    skip()
```

**`fallback()`**

Only meaningful in account-specific rules. Signals "I have no opinion; try the global rules." If global rules also have no decision, the message goes to inbox.

```python
if not my_custom_condition:
    fallback()
```

### Execution Model

1. If the account has custom rules, they run first
2. If those rules call `fallback()` (or make no decision), global rules run
3. If global rules also make no decision or call `fallback()`, the message goes to inbox
4. Any unhandled exception defaults to inbox delivery (fail-safe)

---

## Training Rules

Training rules run when you manually sort a message into a category folder. They update module state so future classification improves.

### Namespace Reference

| Name | Type | Description |
|------|------|-------------|
| `message` | `EmailMessage` | The email being trained on |
| `account` | `str` | Account name |
| `category` | `str` | The destination category (folder name) |
| `modules` | `ModuleNamespace` | Access to loaded modules |

Training rules have no routing functions—there's nothing to route. Call module `.train()` methods to update models.

```python
features = modules.extract_features.classify(message)
modules.naive_bayes.train(message, features, category, account)
modules.match_from.train(message, features, category, account)
```

### Training Guard

Papaya automatically skips training when:

1. The daemon itself just moved the message (tracked via a 5-minute TTL cache)
2. The message was already trained (tracked via `trained_ids.txt`)

You don't need to handle these cases in your rules.

---

## Built-in Modules

Modules expose `classify()` and/or `train()` hooks. Access them via `modules.<name>`.

### `extract_features`

Stateless feature extraction. Call it first to get structured data for ML classifiers.

```python
features = modules.extract_features.classify(message)
```

Returns a `Features` object (see [Data Types](#data-types)).

### `match_from`

Remembers sender→category mappings. Lightning-fast lookups that bypass ML entirely for known senders.

**Classification:**

```python
known = modules.match_from.classify(message, None, account)
if known:
    move_to(known)
```

Returns the category name as a string, or `None` if the sender is unknown.

**Training:**

```python
modules.match_from.train(message, features, category, account)
```

Associates the sender's email address with the category. If the sender was previously in a different category, they're moved.

### `naive_bayes`

Multinomial Naive Bayes classifier. Fast, interpretable, works well with small training sets.

**Classification:**

```python
prediction = modules.naive_bayes.classify(message, features, account)
# prediction.category    → Category enum or None
# prediction.confidence  → float 0.0–1.0
# prediction.scores      → dict mapping Category → score
```

**Training:**

```python
modules.naive_bayes.train(message, features, category, account)
```

### `tfidf_sgd`

TF-IDF vectorisation + SGD (Stochastic Gradient Descent) classifier. Often more accurate than Naive Bayes with sufficient training data.

**Classification:**

```python
prediction = modules.tfidf_sgd.classify(message, features, account)
```

Same return type as `naive_bayes`.

**Training:**

```python
modules.tfidf_sgd.train(message, features, category, account)
```

---

## Data Types

### `Features`

Structured email metadata returned by `extract_features`:

| Field | Type | Description |
|-------|------|-------------|
| `body_text` | `str` | Plain-text body (HTML stripped) |
| `subject` | `str` | Subject line |
| `from_address` | `str` | Sender email address |
| `from_display_name` | `str` | Sender display name |
| `has_list_unsubscribe` | `bool` | True if List-Unsubscribe header present |
| `x_mailer` | `str \| None` | X-Mailer header value |
| `link_count` | `int` | Number of links in body |
| `image_count` | `int` | Number of images |
| `has_form` | `bool` | True if HTML contains form elements |
| `domain_mismatch_score` | `float` | Heuristic for From/Reply-To domain mismatch |
| `is_malformed` | `bool` | True if parsing errors occurred |

### `Prediction`

Classification result from ML modules:

| Field | Type | Description |
|-------|------|-------------|
| `category` | `Category \| None` | Predicted category, or None if untrained |
| `confidence` | `float` | Confidence score (0.0–1.0) |
| `scores` | `Mapping[Category, float]` | Per-category scores |

### `Category`

Enum of known categories:

```python
Category.SPAM         # "Spam"
Category.NEWSLETTERS  # "Newsletters"
Category.IMPORTANT    # "Important"
```

Access the string value with `.value`:

```python
if prediction.category:
    move_to(prediction.category.value)
```

---

## Account-Specific Overrides

Each maildir entry can define its own `rules:` and `train:` blocks:

```yaml
maildirs:
  - name: work
    path: /var/vmail/work.example.com/alice
    rules: |
      # Work-specific classification
      sender = message.get("From", "").lower()
      if "@work.example.com" in sender:
          move_to("Important")
      else:
          fallback()
    train: |
      # Work-specific training (or omit to use global)
      features = modules.extract_features.classify(message)
      modules.naive_bayes.train(message, features, category, account)

  - name: personal
    path: /var/vmail/personal.example.com/alice
    # Uses global rules (no override)
```

### Fallback Semantics

- Account rules that call `move_to()` or `skip()` are final
- Account rules that call `fallback()` (or make no decision) defer to global rules
- Global rules that call `fallback()` (or make no decision) default to inbox

This lets you handle account-specific edge cases while sharing common logic globally.

---

## Real-World Examples

### 1. VIP Sender Whitelist

Route messages from specific senders directly to Important, bypassing ML:

```yaml
rules: |
  vips = [
      "boss@company.com",
      "spouse@family.org",
      "accountant@tax.co",
  ]
  sender = message.get("From", "").lower()
  for vip in vips:
      if vip in sender:
          move_to("Important")
          break
  else:
      # No VIP match; continue with ML
      features = modules.extract_features.classify(message)
      prediction = modules.naive_bayes.classify(message, features, account)
      if prediction.category and prediction.confidence >= 0.6:
          move_to(prediction.category.value, confidence=prediction.confidence)
      else:
          skip()
```

### 2. Newsletter Detection via Headers

Many newsletters include a List-Unsubscribe header. Use it as a strong signal:

```yaml
rules: |
  features = modules.extract_features.classify(message)

  # List-Unsubscribe is a strong newsletter indicator
  if features.has_list_unsubscribe:
      move_to("Newsletters", confidence=0.9)
  else:
      prediction = modules.naive_bayes.classify(message, features, account)
      if prediction.category and prediction.confidence >= 0.55:
          move_to(prediction.category.value, confidence=prediction.confidence)
      else:
          skip()
```

### 3. Domain-Based Routing (Work vs Personal)

Route internal company mail to Important automatically:

```yaml
maildirs:
  - name: work
    path: /var/vmail/example.com/alice
    rules: |
      sender = message.get("From", "").lower()

      # Internal mail is always important
      if "@example.com" in sender or "@subsidiary.example.com" in sender:
          move_to("Important")
      else:
          fallback()  # Let global rules handle external mail
```

### 4. Aggressive Spam Filtering

Use both classifiers and require consensus:

```yaml
rules: |
  features = modules.extract_features.classify(message)

  # Get predictions from both classifiers
  nb = modules.naive_bayes.classify(message, features, account)
  sgd = modules.tfidf_sgd.classify(message, features, account)

  # Aggressive spam: either classifier with high confidence
  if nb.category == Category.SPAM and nb.confidence >= 0.8:
      move_to("Spam", confidence=nb.confidence)
  elif sgd.category == Category.SPAM and sgd.confidence >= 0.8:
      move_to("Spam", confidence=sgd.confidence)
  # Conservative for other categories: require agreement
  elif nb.category and nb.category == sgd.category:
      avg_conf = (nb.confidence + sgd.confidence) / 2
      if avg_conf >= 0.6:
          move_to(nb.category.value, confidence=avg_conf)
      else:
          skip()
  else:
      skip()
```

### 5. Sender Memory + ML Fallback (Recommended Default)

The most practical setup: trust known senders, use ML for unknowns:

```yaml
rules: |
  # Fast path: sender we've seen before
  known = modules.match_from.classify(message, None, account)
  if known:
      move_to(known)
  else:
      # Slow path: ML classification
      features = modules.extract_features.classify(message)
      prediction = modules.naive_bayes.classify(message, features, account)
      if prediction.category and prediction.confidence >= 0.55:
          move_to(prediction.category.value, confidence=prediction.confidence)
      else:
          skip()

train: |
  features = modules.extract_features.classify(message)
  modules.naive_bayes.train(message, features, category, account)
  modules.tfidf_sgd.train(message, features, category, account)
  modules.match_from.train(message, features, category, account)
```

### 6. Subject-Based Rules

Quick filtering based on subject patterns:

```yaml
rules: |
  subject = message.get("Subject", "").lower()

  # Invoice/receipt detection
  if any(word in subject for word in ["invoice", "receipt", "payment confirmed"]):
      move_to("Important")
  # Unsubscribe requests you sent (bounced back)
  elif "unsubscribe" in subject:
      move_to("Newsletters")
  else:
      # Fall through to ML
      features = modules.extract_features.classify(message)
      prediction = modules.naive_bayes.classify(message, features, account)
      if prediction.category and prediction.confidence >= 0.6:
          move_to(prediction.category.value, confidence=prediction.confidence)
      else:
          skip()
```

### 7. Suspicious Link Detection

Flag messages with excessive links as potential spam:

```yaml
rules: |
  features = modules.extract_features.classify(message)

  # Many links + no unsubscribe = suspicious
  if features.link_count > 10 and not features.has_list_unsubscribe:
      move_to("Spam", confidence=0.7)
  # Domain mismatch is a phishing indicator
  elif features.domain_mismatch_score > 0.5:
      move_to("Spam", confidence=0.8)
  else:
      prediction = modules.naive_bayes.classify(message, features, account)
      if prediction.category and prediction.confidence >= 0.55:
          move_to(prediction.category.value, confidence=prediction.confidence)
      else:
          skip()
```

### 8. Conservative Inbox-First Approach

Only sort when very confident; keep uncertain mail in inbox:

```yaml
rules: |
  features = modules.extract_features.classify(message)
  prediction = modules.naive_bayes.classify(message, features, account)

  # Only move with high confidence
  if prediction.category and prediction.confidence >= 0.85:
      move_to(prediction.category.value, confidence=prediction.confidence)
  else:
      skip()  # Stay in inbox for manual review
```

### 9. Logging All Predictions

Track classifier performance for later analysis:

```yaml
rules: |
  features = modules.extract_features.classify(message)

  nb = modules.naive_bayes.classify(message, features, account)
  sgd = modules.tfidf_sgd.classify(message, features, account)

  # Log both predictions
  log_p("naive_bayes", nb)
  log_p("tfidf_sgd", sgd)

  # Use Naive Bayes for routing
  if nb.category and nb.confidence >= 0.6:
      move_to(nb.category.value, confidence=nb.confidence)
  else:
      skip()
```

Logs go to `<root_dir>/logs/predictions.log` in JSON-lines format. Disable this file by setting `logging.write_predictions_logfile: false`.

### 10. Time-Based Rules

Handle mail differently based on headers (requires the `email.utils` module):

```yaml
rules: |
  from email.utils import parsedate_to_datetime

  date_str = message.get("Date")
  if date_str:
      try:
          msg_date = parsedate_to_datetime(date_str)
          # Weekend mail from work domain goes to inbox
          if msg_date.weekday() >= 5:  # Saturday=5, Sunday=6
              sender = message.get("From", "").lower()
              if "@work.example.com" in sender:
                  skip()
                  # Note: can't put code after skip(), so use else
      except Exception:
          pass

  # Default ML path
  features = modules.extract_features.classify(message)
  prediction = modules.naive_bayes.classify(message, features, account)
  if prediction.category and prediction.confidence >= 0.55:
      move_to(prediction.category.value, confidence=prediction.confidence)
  else:
      skip()
```

---

## Error Handling

### Compile-Time Errors

Syntax errors in rules are caught at daemon startup:

```
RuleError: Syntax error in <global_rules>: invalid syntax (line 3)
```

The daemon won't start until you fix the config.

### Runtime Errors

Exceptions during rule execution (e.g., `AttributeError`, `TypeError`) are caught and logged:

```
ERROR Rule execution failed for personal_rules: 'NoneType' object has no attribute 'confidence'
```

The message is delivered to inbox as a fail-safe. Check `<root_dir>/logs/` for details.

### Common Pitfalls

**Forgetting to extract features:**

```python
# Wrong: features not defined
prediction = modules.naive_bayes.classify(message, features, account)

# Right: extract first
features = modules.extract_features.classify(message)
prediction = modules.naive_bayes.classify(message, features, account)
```

**Accessing `.value` on None:**

```python
# Wrong: category might be None
move_to(prediction.category.value)

# Right: check first
if prediction.category:
    move_to(prediction.category.value)
```

**Calling routing functions conditionally without a fallback:**

```python
# Wrong: no decision made if confidence < 0.5
if prediction.confidence >= 0.5:
    move_to(prediction.category.value)
# Message goes to inbox silently

# Better: explicit skip
if prediction.category and prediction.confidence >= 0.5:
    move_to(prediction.category.value)
else:
    skip()
```

---

## Debugging Tips

### 1. Test Classification Manually

```bash
papaya classify /path/to/message.eml -a personal
```

Shows what decision the rules would make without moving the file.

### 2. Check the Prediction Log

```bash
tail -f ~/.local/lib/papaya/logs/predictions.log | jq .
```

Each entry shows the message ID, classifier predictions, and final routing decision. Toggle this via `logging.write_predictions_logfile`.

### 3. Enable Debug Logging

```yaml
logging:
  level: debug
  write_debug_logfile: true
```

Writes verbose logs to `<root_dir>/logs/papaya.log` and enables `<root_dir>/logs/debug.log`.

### 4. Inspect Module State

```bash
papaya status
```

Shows loaded modules, account configurations, and training statistics.

### 5. Reload Without Restart

Send SIGHUP to reload configuration and modules:

```bash
kill -HUP $(cat ~/.local/lib/papaya/papaya.pid)
```

### 6. Dry Run Mode

Test changes without actually moving messages:

```bash
papaya daemon --dry-run
```

The daemon logs what it would do but leaves messages in place.

---

## Quick Reference

### Classification Namespace

```python
message      # EmailMessage object
account      # str: account name
message_id   # str: Message-ID header
modules      # ModuleNamespace: access to modules

move_to(category, confidence=1.0)  # Route to folder
skip()                              # Keep in inbox
fallback()                          # Defer to global rules
log()/log_d()                       # DEBUG-level log helper
log_i()                             # INFO-level log helper
log_p(classifier, prediction)       # Structured prediction log
```

### Training Namespace

```python
message   # EmailMessage object
account   # str: account name
category  # str: destination folder name
modules   # ModuleNamespace: access to modules
log()/log_d()  # DEBUG-level log helper
log_i()        # INFO-level log helper
```

### Module Methods

```python
modules.extract_features.classify(message) → Features
modules.match_from.classify(message, None, account) → str | None
modules.match_from.train(message, features, category, account)
modules.naive_bayes.classify(message, features, account) → Prediction
modules.naive_bayes.train(message, features, category, account)
modules.tfidf_sgd.classify(message, features, account) → Prediction
modules.tfidf_sgd.train(message, features, category, account)
```

---

## Further Reading

- `ARCHITECTURE.md` — system design and data flow diagrams
- `src/papaya/modules/` — built-in module implementations (for writing custom modules)
