# Refactor: Sender Lists → Generic match_from Module

## Summary

Replace the binary whitelist/blacklist sender tracking (`SenderLists`) with a generic per-category `match_from` module. Unify the store API to use pickle for all persistence. Remove `FolderFlag` entirely and let the rule engine handle all sender-based classification.

## Scope

| Remove | Add |
|--------|-----|
| `SenderLists` class | `match_from` module |
| `FolderFlag` enum | Generic `store.get()`/`set()` API |
| `_apply_sender_shortcuts()` | — |
| `save_classifier()`/`load_classifier()` | — |
| `Trainer._apply_sender_flag()` | — |

---

## Phase 1: Generic Store API

**File:** `src/papaya/store.py`

Replace classifier-specific methods with generic pickle-based API:

```python
def get(self, key: str, *, account: str | None = None) -> Any | None:
    """Load a pickled object by key. Returns None if missing/corrupt."""

def set(self, key: str, value: Any, *, account: str | None = None) -> Path:
    """Persist a Python object using atomic pickle write."""
```

- Storage path: `<root>/data/<account|global>/<key>.pkl`
- Reuse `_atomic_write()` for atomic temp-file-and-rename
- Add `_quarantine_corrupt_file()` handling for corrupt pickles
- Delete `save_classifier()`, `load_classifier()`, `model_path()`
- Remove `Classifier` import and protocol dependency

**Tests:** `tests/unit/test_store.py`
- Round-trip pickle serialization
- Missing key returns `None`
- Corrupt pickle is quarantined
- Account vs global namespace separation

---

## Phase 2: Update Existing Classifier Modules

**Files:** `src/papaya/modules/naive_bayes.py`, `src/papaya/modules/tfidf_sgd.py`

Update to use new generic store API. The classifiers need to change their persistence pattern:

```python
# Before
_STORE.save_classifier(model, account=account)
_STORE.load_classifier(model, account=account)

# After
_STORE.set(model.name, model, account=account)
model_data = _STORE.get(model.name, account=account)
if model_data is not None:
    model = model_data  # Already deserialized
```

This means classifiers are pickled directly rather than delegating to their own `save()`/`load()` methods. The classifier `Classifier` protocol's save/load methods become unused (can remove in future cleanup).

**Tests:** Update `tests/unit/test_builtin_modules.py` if it tests persistence behaviour.

---

## Phase 3: Create match_from Module

**File:** `src/papaya/modules/match_from.py`

### Data Structure

```python
# Per-account: category_name → set of normalized "from" addresses
_ADDRESSES: dict[str, dict[str, set[str]]] = {}  # account → {category → {addr}}
```

### Lifecycle Hooks

**`startup(ctx: ModuleContext)`:**
1. Store reference to `ctx.store`
2. For each account in `ctx.config.maildirs`:
   - Try `store.get("match_from", account=account.name)`
   - If missing or `ctx.fresh_models`: scan category folders to rebuild
   - Cache in `_ADDRESSES[account]`

**`cleanup()`:**
- Persist all cached address sets via `store.set()`
- Clear module globals

### Public API

**`classify(message, features, account) -> str | None`:**
- Extract and normalise "From" address
- Return category name if address found in any category set, else `None`

**`train(message, features, category, account) -> None`:**
- Extract and normalise "From" address
- Remove address from all other category sets (handles re-categorisation)
- Add address to specified category set
- Persist updated state

### Thread Safety

- Use `threading.Lock()` around all reads/writes to `_ADDRESSES`
- Atomic file writes via `Store.set()`

**Tests:** `tests/unit/modules/test_match_from.py`
- `test_startup_loads_persisted_addresses`
- `test_startup_scans_folders_when_fresh`
- `test_classify_returns_matching_category`
- `test_classify_returns_none_when_unknown`
- `test_train_adds_address_to_category`
- `test_train_removes_from_previous_category`
- `test_cleanup_persists_state`

---

## Phase 4: Remove FolderFlag

**Files to modify:**

| File | Change |
|------|--------|
| `src/papaya/types.py` | Delete `FolderFlag` enum |
| `src/papaya/types.py` | Remove `flag` field from `CategoryConfig` |
| `src/papaya/config.py` | Delete `_parse_flag()` function |
| `src/papaya/config.py` | Remove `flag=` from `CategoryConfig` creation |
| All test files | Update `CategoryConfig` instantiations to drop `flag=` |

---

## Phase 5: Remove Sender Shortcuts from Runtime

**File:** `src/papaya/runtime.py`

Delete:
- `_apply_sender_shortcuts()` method
- `_deliver_by_flag()` method
- `_build_flag_targets()` helper
- `self.senders` field from `AccountRuntime`
- `self._flag_targets` from `__post_init__`

Remove from `ClassificationMetrics`:
- `whitelist_hits: int`
- `blacklist_hits: int`

Simplify `_handle_new_mail()`:
```python
# DELETE THIS BLOCK:
if self._apply_sender_shortcuts(target, message_id, from_address):
    return
```

---

## Phase 6: Update Trainer

**File:** `src/papaya/trainer.py`

- Remove `from .senders import SenderLists` import
- Remove `senders` parameter from `__init__`
- Remove `self._senders` field
- Delete `_apply_sender_flag()` method
- Remove call to `self._apply_sender_flag(config, message)` in `on_user_sort()`

The module's `train()` hook handles sender tracking now via rule engine's train rules.

---

## Phase 7: Delete SenderLists

- Delete `src/papaya/senders.py`
- Delete `tests/unit/test_senders.py`
- Remove all imports of `SenderLists` from other files

---

## Phase 8: Update CLI Integration

**File:** `src/papaya/cli.py`

- Delete `senders = SenderLists(config.root_dir)`
- Update `Trainer()` construction to remove `senders=` parameter
- Remove `senders` from `_build_account_runtime()` if present
- Update reload callback if it handles senders

---

## Phase 9: Update Default Rules (Documentation)

Document the new pattern for users. Classification rules should call `match_from`:

```python
# Classification rules
known_category = modules.match_from.classify(message, None, account)
if known_category:
    move_to(known_category)
else:
    features = modules.extract_features.classify(message)
    prediction = modules.naive_bayes.classify(message, features, account)
    if prediction.category and prediction.confidence >= 0.55:
        move_to(prediction.category.value, confidence=prediction.confidence)
    else:
        skip()
```

Training rules should include `match_from.train()`:

```python
# Training rules
features = modules.extract_features.classify(message)
modules.naive_bayes.train(message, features, category, account)
modules.match_from.train(message, features, category, account)
```

---

## Implementation Order

Minimise broken intermediate states:

1. **Phase 1** — Generic store API (additive)
2. **Phase 2** — Update classifier modules to new API
3. **Phase 3** — Create match_from module (additive)
4. **Phase 4** — Remove FolderFlag (update all tests simultaneously)
5. **Phase 5** — Remove sender shortcuts from AccountRuntime
6. **Phase 6** — Update Trainer
7. **Phase 7** — Delete SenderLists
8. **Phase 8** — Update CLI integration
9. **Phase 9** — Update docs/example rules

**Commits:**
1. `feat: Add generic key/value store API`
2. `refactor: Update classifier modules to use generic store`
3. `feat: Add match_from module for per-category sender tracking`
4. `refactor: Remove FolderFlag enum and CategoryConfig.flag`
5. `refactor: Remove sender shortcuts from AccountRuntime`
6. `refactor: Remove sender handling from Trainer`
7. `chore: Delete SenderLists and related tests`

---

## Critical Files

Files to read before implementation:
- `src/papaya/store.py` — current persistence layer
- `src/papaya/senders.py` — existing implementation to replace
- `src/papaya/runtime.py` — sender shortcut integration points
- `src/papaya/trainer.py` — training flow
- `src/papaya/types.py` — FolderFlag and CategoryConfig
- `src/papaya/modules/naive_bayes.py` — module pattern reference
- `src/papaya/config.py` — flag parsing

---

## Edge Cases

1. **Address normalisation**: Handle `"Display Name" <email@example.com>` format
2. **Category transitions**: When user moves message, remove address from old category before adding to new
3. **Cold start**: First run scans folders — could be slow for large maildirs
4. **Concurrent threads**: Module lock prevents race conditions
5. **Pickle versioning**: Handle gracefully if structure changes (fall back to scan)
6. **Empty From headers**: Return `None` from classify, skip in train
