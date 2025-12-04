from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from papaya import types as papaya_types


def test_prediction_uses_strings() -> None:
    prediction = papaya_types.Prediction(category="Spam", confidence=0.85, scores={"Spam": 0.85})
    assert prediction.category == "Spam"
    assert prediction.scores["Spam"] == pytest.approx(0.85)


def test_features_is_immutable() -> None:
    features = papaya_types.Features(
        body_text="body",
        subject="subject",
        from_address="sender@example.com",
        from_display_name="Sender",
        has_list_unsubscribe=False,
        x_mailer=None,
        link_count=0,
        image_count=0,
        has_form=False,
        domain_mismatch_score=0.0,
        is_malformed=False,
    )

    with pytest.raises(FrozenInstanceError):
        features.body_text = "new"  # type: ignore[misc]


def test_maildir_account_path_expansion() -> None:
    account = papaya_types.MaildirAccount(name="inbox", path=Path("/tmp/maildir"))
    assert account.name == "inbox"
    assert account.path == Path("/tmp/maildir")


def test_category_config_structure() -> None:
    cfg = papaya_types.CategoryConfig(name="Spam")
    assert cfg.name == "Spam"
