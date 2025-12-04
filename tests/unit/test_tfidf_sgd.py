from __future__ import annotations

from pathlib import Path

from papaya.classifiers.tfidf_sgd import TfidfSgdClassifier
from papaya.types import Features


def _sample_features(subject: str, body: str) -> Features:
    return Features(
        body_text=body,
        subject=subject,
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


def test_tfidf_sgd_trains_and_predicts(tmp_path: Path) -> None:
    classifier = TfidfSgdClassifier(random_state=7)
    spam = _sample_features("Win millions", "Lottery winner claim prize now now now")
    ham = _sample_features("Sprint Planning", "Agenda stories backlog discussion planning session")

    for _ in range(40):
        classifier.train(spam, "Spam")
        classifier.train(ham, "Important")

    assert classifier.is_trained() is True

    spam_prediction = classifier.predict(spam)
    ham_prediction = classifier.predict(ham)

    assert spam_prediction.category == "Spam"
    assert ham_prediction.category == "Important"

    model_path = tmp_path / "tfidf.pkl"
    classifier.save(model_path)

    restored = TfidfSgdClassifier()
    restored.load(model_path)
    restored_prediction = restored.predict(ham)

    assert restored_prediction.category == "Important"


def test_tfidf_sgd_handles_untrained_state() -> None:
    classifier = TfidfSgdClassifier()
    features = _sample_features("Newsletter", "Monthly digest and update")
    prediction = classifier.predict(features)
    assert prediction.category is None
    assert prediction.confidence == 0.0
