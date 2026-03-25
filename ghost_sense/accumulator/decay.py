"""Continuous exponential decay — applied at read-time, never by cron.

Core formula: confidence_now = confidence_then * e^(-lambda * delta_hours)

All decay is lazy: stored values are raw (undecayed). Decay is computed
on every read against wall-clock time since last update.
"""

from __future__ import annotations

import math

from ghost_sense.config import get_lambda


def apply_decay(confidence: float, hours_elapsed: float, lam: float) -> float:
    """Apply exponential decay to a confidence value.

    Args:
        confidence: The original confidence (0.0–1.0).
        hours_elapsed: Hours since the confidence was last updated.
        lam: Decay rate (higher = faster decay).

    Returns:
        Decayed confidence. Always in [0.0, confidence].
    """
    if hours_elapsed <= 0 or lam <= 0:
        return confidence
    return confidence * math.exp(-lam * hours_elapsed)


def apply_decay_for_dimension(confidence: float, hours_elapsed: float, dimension: str) -> float:
    """Apply decay using the configured lambda for a specific dimension."""
    return apply_decay(confidence, hours_elapsed, get_lambda(dimension))


def half_life_hours(lam: float) -> float:
    """Compute the half-life in hours for a given lambda.

    Half-life = ln(2) / lambda
    """
    if lam <= 0:
        return float("inf")
    return math.log(2) / lam


def hours_until_threshold(confidence: float, threshold: float, lam: float) -> float:
    """How many hours until confidence decays below a threshold.

    Returns inf if confidence is already below threshold or lambda is 0.
    """
    if confidence <= threshold or lam <= 0:
        return 0.0 if confidence <= threshold else float("inf")
    # threshold = confidence * e^(-lam * t)  =>  t = -ln(threshold/confidence) / lam
    return -math.log(threshold / confidence) / lam


def merge_with_decay(
    old_value: float,
    old_confidence: float,
    new_value: float,
    new_confidence: float,
    hours_elapsed: float,
    lam: float,
) -> tuple[float, float]:
    """Merge a new observation with a decayed old observation.

    Returns (merged_value, merged_confidence).
    The new signal is weighted by its confidence, the old signal by its
    decayed confidence. This naturally favors recent observations.
    """
    decayed_old = apply_decay(old_confidence, hours_elapsed, lam)
    total = decayed_old + new_confidence
    if total <= 0:
        return new_value, new_confidence
    merged_value = (old_value * decayed_old + new_value * new_confidence) / total
    merged_confidence = min(total, 1.0)
    return merged_value, merged_confidence
