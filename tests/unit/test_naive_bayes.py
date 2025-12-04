from __future__ import annotations

from pathlib import Path

from papaya.classifiers.naive_bayes import NaiveBayesClassifier
from papaya.types import Features


def _sample_features(subject: str, body: str) -> Features:
    return Features(
        body_text=body,
        subject=subject,
        from_address="sender@example.com",
        from_display_name="Sender",
        has_list_unsubscribe=False,
        x_mailer="MailerX",
        link_count=0,
        image_count=0,
        has_form=False,
        domain_mismatch_score=0.0,
        is_malformed=False,
    )


def test_naive_bayes_trains_and_predicts(tmp_path: Path) -> None:
    classifier = NaiveBayesClassifier()
    spam = _sample_features("Win now", "Cheap meds cheap pills free money")
    ham = _sample_features("Project Update", "Agenda project meeting schedule discussion")

    for _ in range(20):
        classifier.train(spam, "Spam")
        classifier.train(ham, "Important")

    assert classifier.is_trained() is True

    spam_prediction = classifier.predict(spam)
    ham_prediction = classifier.predict(ham)

    assert spam_prediction.category == "Spam"
    assert ham_prediction.category == "Important"
    assert 0.0 <= spam_prediction.confidence <= 1.0
    assert 0.0 <= ham_prediction.confidence <= 1.0

    model_path = tmp_path / "naive_bayes.pkl"
    classifier.save(model_path)
    assert model_path.exists()

    restored = NaiveBayesClassifier()
    restored.load(model_path)

    restored_prediction = restored.predict(spam)
    assert restored_prediction.category == spam_prediction.category


def test_naive_bayes_returns_empty_prediction_when_untrained() -> None:
    classifier = NaiveBayesClassifier()
    features = _sample_features("Hello", "Welcome to the team")
    prediction = classifier.predict(features)
    assert prediction.category is None
    assert prediction.confidence == 0.0
