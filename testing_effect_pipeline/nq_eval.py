"""NQ-style answer normalization and exact-match scoring."""

from __future__ import annotations

import re
import string


def normalize_nq_answer(text: str) -> str:
    """Normalize answer for NQ exact-match evaluation.

    Standard pipeline: lowercase -> remove articles -> remove punctuation -> collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, targets: list[str]) -> bool:
    """True if normalized prediction matches any normalized target."""
    norm_pred = normalize_nq_answer(prediction)
    return any(normalize_nq_answer(t) == norm_pred for t in targets)
