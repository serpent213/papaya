from __future__ import annotations

import pytest

from papaya.classifiers.registry import ClassifierRegistry
from papaya.types import Category, ClassifierMode, Features, Prediction


class DummyClassifier:
    def __init__(self, name: str, predicted: Category) -> None:
        self.name = name
        self.predicted = predicted
        self.trained_labels: list[Category] = []

    def train(self, _features: Features, label: Category) -> None:
        self.trained_labels.append(label)

    def predict(self, _features: Features) -> Prediction:
        return Prediction(
            category=self.predicted,
            confidence=1.0,
            scores={self.predicted: 1.0},
        )

    def save(self, path):  # pragma: no cover - not used in registry tests
        pass

    def load(self, path):  # pragma: no cover - not used in registry tests
        pass

    def is_trained(self) -> bool:
        return bool(self.trained_labels)


def _sample_features() -> Features:
    return Features(
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


def test_registry_tracks_active_and_shadow_classifiers() -> None:
    registry = ClassifierRegistry()
    active = DummyClassifier("active", Category.SPAM)
    shadow = DummyClassifier("shadow", Category.NEWSLETTERS)

    registry.register(active, ClassifierMode.ACTIVE)
    registry.register(shadow, ClassifierMode.SHADOW)

    assert registry.get_active() is active
    registry.train_all(_sample_features(), Category.SPAM)

    assert active.trained_labels == [Category.SPAM]
    assert shadow.trained_labels == [Category.SPAM]

    predictions = registry.predict_all(_sample_features())
    assert predictions["active"].category == Category.SPAM
    assert predictions["shadow"].category == Category.NEWSLETTERS


def test_registry_rejects_duplicate_names() -> None:
    registry = ClassifierRegistry()
    classifier = DummyClassifier("dup", Category.SPAM)
    registry.register(classifier, ClassifierMode.ACTIVE)
    with pytest.raises(ValueError):
        registry.register(classifier, ClassifierMode.SHADOW)
