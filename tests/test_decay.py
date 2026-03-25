"""Tests for decay math — pure functions with well-defined properties."""

import math

import pytest

from ghost_sense.accumulator.decay import (
    apply_decay,
    apply_decay_for_dimension,
    half_life_hours,
    hours_until_threshold,
    merge_with_decay,
)
from ghost_sense.config import CONFIDENCE_THRESHOLD, get_lambda


class TestApplyDecay:
    def test_no_time_elapsed_no_decay(self):
        assert apply_decay(1.0, 0.0, 0.5) == 1.0

    def test_zero_confidence_stays_zero(self):
        assert apply_decay(0.0, 10.0, 0.5) == 0.0

    def test_decay_reduces_confidence(self):
        result = apply_decay(1.0, 2.0, 0.5)
        assert 0.0 < result < 1.0

    def test_higher_lambda_faster_decay(self):
        slow = apply_decay(1.0, 5.0, 0.1)
        fast = apply_decay(1.0, 5.0, 0.5)
        assert fast < slow

    def test_longer_time_more_decay(self):
        short = apply_decay(1.0, 1.0, 0.3)
        long = apply_decay(1.0, 10.0, 0.3)
        assert long < short

    def test_approaches_zero_over_time(self):
        result = apply_decay(1.0, 100.0, 0.5)
        assert result < 0.001

    def test_never_goes_negative(self):
        result = apply_decay(1.0, 10000.0, 10.0)
        assert result >= 0.0

    def test_negative_time_no_decay(self):
        """Negative elapsed time = no decay (clock skew protection)."""
        assert apply_decay(0.8, -5.0, 0.5) == 0.8

    def test_zero_lambda_no_decay(self):
        assert apply_decay(0.8, 100.0, 0.0) == 0.8

    def test_monotonic_decrease(self):
        """Confidence must monotonically decrease with time."""
        prev = 1.0
        for hours in range(1, 50):
            current = apply_decay(1.0, float(hours), 0.3)
            assert current <= prev
            prev = current

    def test_exact_math(self):
        """Verify against hand-computed value."""
        # e^(-0.5 * 2) = e^(-1) ≈ 0.36788
        result = apply_decay(1.0, 2.0, 0.5)
        assert abs(result - math.exp(-1.0)) < 1e-10

    def test_scales_linearly_with_initial_confidence(self):
        r1 = apply_decay(1.0, 3.0, 0.2)
        r2 = apply_decay(0.5, 3.0, 0.2)
        assert abs(r2 - r1 * 0.5) < 1e-10


class TestApplyDecayForDimension:
    def test_uses_configured_lambda(self):
        dim = "register.slang_ratio"
        lam = get_lambda(dim)
        expected = apply_decay(0.9, 5.0, lam)
        result = apply_decay_for_dimension(0.9, 5.0, dim)
        assert abs(result - expected) < 1e-10

    def test_unknown_dimension_uses_default(self):
        from ghost_sense.config import DEFAULT_DECAY_LAMBDA
        result = apply_decay_for_dimension(1.0, 5.0, "unknown.dimension")
        expected = apply_decay(1.0, 5.0, DEFAULT_DECAY_LAMBDA)
        assert abs(result - expected) < 1e-10

    def test_formality_decays_slower_than_slang(self):
        formality = apply_decay_for_dimension(1.0, 10.0, "register.formality_score")
        slang = apply_decay_for_dimension(1.0, 10.0, "register.slang_ratio")
        assert formality > slang

    def test_sleep_decays_slower_than_register(self):
        sleep = apply_decay_for_dimension(1.0, 24.0, "sleep.quality")
        register = apply_decay_for_dimension(1.0, 24.0, "register.slang_ratio")
        assert sleep > register


class TestHalfLife:
    def test_known_half_life(self):
        """lambda=0.5 -> half-life = ln(2)/0.5 ≈ 1.386h"""
        hl = half_life_hours(0.5)
        assert abs(hl - math.log(2) / 0.5) < 1e-10

    def test_zero_lambda_infinite_half_life(self):
        assert half_life_hours(0.0) == float("inf")

    def test_higher_lambda_shorter_half_life(self):
        assert half_life_hours(1.0) < half_life_hours(0.1)

    def test_at_half_life_confidence_is_half(self):
        lam = 0.3
        hl = half_life_hours(lam)
        result = apply_decay(1.0, hl, lam)
        assert abs(result - 0.5) < 1e-10


class TestHoursUntilThreshold:
    def test_already_below_returns_zero(self):
        assert hours_until_threshold(0.1, 0.5, 0.3) == 0.0

    def test_zero_lambda_returns_inf(self):
        assert hours_until_threshold(1.0, 0.5, 0.0) == float("inf")

    def test_known_answer(self):
        # 0.15 = 1.0 * e^(-0.3 * t) => t = -ln(0.15)/0.3
        expected = -math.log(0.15) / 0.3
        result = hours_until_threshold(1.0, 0.15, 0.3)
        assert abs(result - expected) < 1e-10

    def test_consistency_with_decay(self):
        """Decaying for exactly the predicted hours should hit the threshold."""
        hours = hours_until_threshold(0.9, CONFIDENCE_THRESHOLD, 0.2)
        result = apply_decay(0.9, hours, 0.2)
        assert abs(result - CONFIDENCE_THRESHOLD) < 1e-8


class TestMergeWithDecay:
    def test_no_old_data(self):
        """When old confidence decays to ~0, new signal dominates."""
        val, conf = merge_with_decay(0.5, 0.01, 0.8, 0.9, 1000.0, 0.5)
        assert abs(val - 0.8) < 0.01
        assert conf > 0.8

    def test_equal_weight_averages(self):
        """When old and new have equal confidence and no decay, value is average."""
        val, conf = merge_with_decay(0.2, 0.5, 0.8, 0.5, 0.0, 0.3)
        assert abs(val - 0.5) < 1e-10

    def test_new_signal_wins_after_decay(self):
        """After significant decay, new signal should dominate."""
        val, conf = merge_with_decay(0.3, 0.9, 0.7, 0.9, 20.0, 0.5)
        assert val > 0.5  # closer to new (0.7) than old (0.3)

    def test_confidence_capped_at_one(self):
        _, conf = merge_with_decay(0.5, 0.9, 0.5, 0.9, 0.0, 0.1)
        assert conf <= 1.0

    def test_merged_value_between_inputs(self):
        val, _ = merge_with_decay(0.2, 0.8, 0.9, 0.8, 1.0, 0.3)
        assert 0.2 <= val <= 0.9
