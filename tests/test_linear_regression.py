"""Unit tests for the pure OLS helper used by trend detection.

These exercise ``_linear_regression`` directly (no SQLite, no I/O), covering
the perfect-fit case and the degenerate-input guards that protect callers
from division-by-zero and empty-sequence errors.
"""

import math

from ghost_sense.accumulator.trend import _linear_regression


class TestPerfectFit:
    def test_recovers_known_line(self):
        # y = 2x + 1
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 3.0, 5.0, 7.0, 9.0]
        slope, intercept = _linear_regression(xs, ys)
        assert math.isclose(slope, 2.0, abs_tol=1e-9)
        assert math.isclose(intercept, 1.0, abs_tol=1e-9)

    def test_negative_slope(self):
        # y = -0.5x + 10
        xs = [0.0, 2.0, 4.0, 6.0]
        ys = [10.0, 9.0, 8.0, 7.0]
        slope, intercept = _linear_regression(xs, ys)
        assert math.isclose(slope, -0.5, abs_tol=1e-9)
        assert math.isclose(intercept, 10.0, abs_tol=1e-9)

    def test_flat_line_zero_slope(self):
        slope, intercept = _linear_regression([0.0, 1.0, 2.0, 3.0], [5.0, 5.0, 5.0, 5.0])
        assert math.isclose(slope, 0.0, abs_tol=1e-12)
        assert math.isclose(intercept, 5.0, abs_tol=1e-9)


class TestBestFitThroughNoise:
    def test_slope_sign_matches_trend(self):
        # Rising but noisy — slope should still be positive.
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.1, 0.3, 0.2, 0.5, 0.6]
        slope, _ = _linear_regression(xs, ys)
        assert slope > 0

    def test_intercept_is_mean_offset(self):
        # For a centered, symmetric x with a known slope, intercept equals
        # mean(y) - slope * mean(x).
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 5.0, 9.0]
        slope, intercept = _linear_regression(xs, ys)
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        assert math.isclose(intercept, mean_y - slope * mean_x, rel_tol=1e-9)


class TestDegenerateInputs:
    def test_single_point_zero_slope(self):
        # Cannot fit a slope to one point; intercept is that point's y.
        slope, intercept = _linear_regression([4.0], [9.0])
        assert slope == 0.0
        assert intercept == 9.0

    def test_empty_returns_zeros(self):
        slope, intercept = _linear_regression([], [])
        assert slope == 0.0
        assert intercept == 0.0

    def test_identical_x_no_divide_by_zero(self):
        # All xs equal => zero variance in x => guarded denominator.
        # Should not raise; slope falls back to 0 and intercept to mean(y).
        slope, intercept = _linear_regression([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
        assert slope == 0.0
        assert math.isclose(intercept, 2.0, abs_tol=1e-9)
