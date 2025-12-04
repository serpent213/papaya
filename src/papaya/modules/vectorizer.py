"""Feature vectorization module."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from papaya.modules.context import ModuleContext
from papaya.types import Features

NUMERIC_FEATURE_DIM = 8
DEFAULT_TEXT_DIM = 2**14


@dataclass(frozen=True)
class EncodedFeatures:
    """Sparse matrices for text and numeric features."""

    text: sparse.csr_matrix
    numeric: sparse.csr_matrix


class FeatureVectoriser:
    """Transforms structured features into sparse matrices."""

    def __init__(self, text_features: int = DEFAULT_TEXT_DIM) -> None:
        self._text_features = text_features
        self._vectorizer = HashingVectorizer(
            n_features=text_features,
            alternate_sign=False,
            norm=None,
            binary=False,
            lowercase=True,
            stop_words=None,
        )

    @property
    def text_dimension(self) -> int:
        return self._text_features

    @property
    def numeric_dimension(self) -> int:
        return NUMERIC_FEATURE_DIM

    def transform(self, features: Features) -> EncodedFeatures:
        """Encode a Features object into sparse matrices."""

        document = self._build_document(features)
        text_matrix = self._vectorizer.transform([document])
        numeric_values = np.asarray([_numeric_features(features)], dtype=np.float64)
        numeric_matrix = sparse.csr_matrix(numeric_values)
        return EncodedFeatures(text=text_matrix, numeric=numeric_matrix)

    def _build_document(self, features: Features) -> str:
        tokens: list[str] = [
            features.subject,
            features.body_text,
            features.from_address,
            features.from_display_name,
        ]
        if features.x_mailer:
            tokens.append(features.x_mailer)
        if features.has_form:
            tokens.append("flag:has_form")
        if features.has_list_unsubscribe:
            tokens.append("flag:list_unsubscribe")
        if features.is_malformed:
            tokens.append("flag:malformed")
        tokens.append(f"domain_mismatch:{int(features.domain_mismatch_score)}")
        return "\n".join(part for part in tokens if part)


def _numeric_features(features: Features) -> list[float]:
    return [
        math.log1p(features.link_count),
        math.log1p(features.image_count),
        1.0 if features.has_form else 0.0,
        float(features.domain_mismatch_score),
        1.0 if features.is_malformed else 0.0,
        1.0 if features.has_list_unsubscribe else 0.0,
        math.log1p(len(features.subject)),
        math.log1p(len(features.body_text)),
    ]


_VECTORISER: FeatureVectoriser | None = None


def startup(_ctx: ModuleContext) -> None:
    """Initialise the shared feature vectoriser."""
    global _VECTORISER
    _VECTORISER = FeatureVectoriser()


def cleanup() -> None:
    """Reset the shared vectoriser."""
    global _VECTORISER
    _VECTORISER = None


def get_vectoriser() -> FeatureVectoriser:
    """Return the shared FeatureVectoriser instance."""
    if _VECTORISER is None:
        raise RuntimeError("vectorizer module not initialized")
    return _VECTORISER


__all__ = [
    "startup",
    "cleanup",
    "get_vectoriser",
    "FeatureVectoriser",
    "EncodedFeatures",
    "DEFAULT_TEXT_DIM",
    "NUMERIC_FEATURE_DIM",
]
