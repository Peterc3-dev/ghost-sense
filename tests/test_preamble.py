"""Tests for preamble generation and adapter output format."""

import pytest

from ghost_sense.accumulator.trend import Trend, TrendDirection
from ghost_sense.conditioner.adapters.boo2 import format_boo2_system_message, format_boo2_with_base_prompt
from ghost_sense.conditioner.adapters.claude import format_claude_api_messages, format_claude_system_prompt
from ghost_sense.conditioner.preamble import (
    PREAMBLE_FOOTER,
    PREAMBLE_HEADER,
    generate_preamble,
)
from ghost_sense.models import FieldState, SignalType, StateVector


def _make_state(**fields) -> StateVector:
    """Helper: build a StateVector from dimension=value pairs."""
    sv = StateVector()
    for dim, val in fields.items():
        sig_type = dim.split(".")[0]
        sv.set(FieldState(
            dimension=dim,
            signal_type=SignalType(sig_type),
            value=val,
            confidence=0.85,
            last_updated=1000000.0,
        ))
    return sv


def _tired_state() -> StateVector:
    return _make_state(
        **{
            "sleep.quality": 0.2,
            "sleep.energy_level": 0.2,
            "sleep.caffeine_proxy": 0.3,
            "stress.level": 0.7,
            "stress.source": 1.0,
            "register.formality_score": 0.3,
            "register.slang_ratio": 0.5,
            "cadence.burst_score": 0.8,
            "nutrition.last_meal": 0.0,
        }
    )


def _fresh_state() -> StateVector:
    return _make_state(
        **{
            "sleep.quality": 0.9,
            "sleep.energy_level": 0.9,
            "stress.level": 0.15,
            "register.formality_score": 0.8,
            "register.slang_ratio": 0.02,
            "cadence.burst_score": 0.2,
            "nutrition.last_meal": 1.0,
            "nutrition.quality": 0.85,
        }
    )


class TestPreambleFormat:
    def test_has_header_and_footer(self):
        preamble = generate_preamble(_tired_state())
        assert preamble.startswith(PREAMBLE_HEADER)
        assert preamble.endswith(PREAMBLE_FOOTER)

    def test_contains_state_section(self):
        preamble = generate_preamble(_tired_state())
        assert "User state estimate" in preamble

    def test_contains_adaptation_section(self):
        preamble = generate_preamble(_tired_state())
        assert "Suggested adaptation:" in preamble

    def test_empty_state_still_valid(self):
        preamble = generate_preamble(StateVector())
        assert PREAMBLE_HEADER in preamble
        assert PREAMBLE_FOOTER in preamble
        assert "No specific adaptations" in preamble


class TestPreambleNeverRevealsAssessment:
    """Critical: the preamble must NEVER contain user-facing language."""

    FORBIDDEN_PHRASES = [
        "I notice",
        "you seem",
        "you look",
        "you appear",
        "are you okay",
        "are you tired",
        "how are you feeling",
        "you should sleep",
        "you should eat",
        "take a break",
        "you need rest",
        "I can tell",
        "it looks like you",
        "it seems like you",
        "based on your messages",
        "I've been monitoring",
        "I've been tracking",
        "I've noticed",
        "your wellbeing",
        "your well-being",
        "your health",
        "assessment shows",
        "analysis indicates",
    ]

    def test_tired_state_no_reveal(self):
        preamble = generate_preamble(_tired_state())
        lower = preamble.lower()
        for phrase in self.FORBIDDEN_PHRASES:
            assert phrase.lower() not in lower, f"Forbidden phrase found: '{phrase}'"

    def test_fresh_state_no_reveal(self):
        preamble = generate_preamble(_fresh_state())
        lower = preamble.lower()
        for phrase in self.FORBIDDEN_PHRASES:
            assert phrase.lower() not in lower, f"Forbidden phrase found: '{phrase}'"

    def test_with_trends_no_reveal(self):
        trends = {
            "sleep.quality": {
                3: Trend("sleep.quality", 3, TrendDirection.FALLING, 0.05, -0.05, 5),
            }
        }
        preamble = generate_preamble(_tired_state(), trends=trends)
        lower = preamble.lower()
        for phrase in self.FORBIDDEN_PHRASES:
            assert phrase.lower() not in lower, f"Forbidden phrase found: '{phrase}'"

    def test_only_contains_directives(self):
        """Adaptation lines should be directives (imperative), not observations."""
        preamble = generate_preamble(_tired_state())
        lines = preamble.split("\n")
        adaptation_lines = []
        in_adaptation = False
        for line in lines:
            if "Suggested adaptation:" in line:
                in_adaptation = True
                continue
            if in_adaptation and line.startswith("- "):
                adaptation_lines.append(line)
            elif in_adaptation and line == PREAMBLE_FOOTER:
                break
        # All adaptation lines should be imperative directives
        for line in adaptation_lines:
            stripped = line.lstrip("- ").strip()
            # Should not start with "You" (second person observation)
            assert not stripped.startswith("You "), f"Adaptation should be directive, not observation: {line}"


class TestPreambleContent:
    def test_sleep_deficit_detected(self):
        preamble = generate_preamble(_tired_state())
        assert "deficit" in preamble.lower() or "low energy" in preamble.lower()

    def test_high_stress_detected(self):
        preamble = generate_preamble(_tired_state())
        assert "elevated" in preamble.lower() or "stress" in preamble.lower()

    def test_casual_register_detected(self):
        preamble = generate_preamble(_tired_state())
        assert "casual" in preamble.lower()

    def test_formal_register_detected(self):
        preamble = generate_preamble(_fresh_state())
        assert "formal" in preamble.lower()

    def test_hunger_detected(self):
        preamble = generate_preamble(_tired_state())
        assert "hunger" in preamble.lower() or "skipping" in preamble.lower()

    def test_burst_cadence_detected(self):
        preamble = generate_preamble(_tired_state())
        assert "rapid" in preamble.lower()

    def test_cognitive_load_adaptation_when_tired(self):
        preamble = generate_preamble(_tired_state())
        assert "concise" in preamble.lower() or "cognitive load" in preamble.lower()

    def test_late_night_circadian(self):
        preamble = generate_preamble(_tired_state(), local_time="02:30")
        assert "late" in preamble.lower() and "02:30" in preamble

    def test_no_circadian_midday(self):
        preamble = generate_preamble(_tired_state(), local_time="14:00")
        assert "circadian" not in preamble.lower()


class TestPreambleTrends:
    def test_trend_section_included(self):
        trends = {
            "sleep.quality": {
                3: Trend("sleep.quality", 3, TrendDirection.FALLING, 0.05, -0.05, 5),
                7: Trend("sleep.quality", 7, TrendDirection.STABLE, 0.002, 0.002, 10),
            }
        }
        preamble = generate_preamble(_tired_state(), trends=trends)
        assert "Trends:" in preamble
        assert "trending down" in preamble.lower()

    def test_insignificant_trends_excluded(self):
        trends = {
            "sleep.quality": {
                3: Trend("sleep.quality", 3, TrendDirection.STABLE, 0.001, 0.001, 2),
            }
        }
        preamble = generate_preamble(_tired_state(), trends=trends)
        assert "Trends:" not in preamble


class TestClaudeAdapter:
    def test_system_prompt_prepend(self):
        result = format_claude_system_prompt("You are a helpful assistant.", _tired_state())
        assert result.startswith(PREAMBLE_HEADER)
        assert "You are a helpful assistant." in result

    def test_empty_base_prompt(self):
        result = format_claude_system_prompt("", _tired_state())
        assert result.startswith(PREAMBLE_HEADER)
        assert result.endswith(PREAMBLE_FOOTER)

    def test_api_message_format(self):
        result = format_claude_api_messages(_tired_state())
        assert result["type"] == "text"
        assert PREAMBLE_HEADER in result["text"]
        assert result["cache_control"] == {"type": "ephemeral"}


class TestBoo2Adapter:
    def test_system_message_format(self):
        result = format_boo2_system_message(_tired_state())
        assert result["role"] == "system"
        assert PREAMBLE_HEADER in result["content"]

    def test_with_base_prompt(self):
        result = format_boo2_with_base_prompt("Base instructions.", _tired_state())
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert PREAMBLE_HEADER in result[0]["content"]
        assert result[1]["content"] == "Base instructions."

    def test_empty_base_prompt(self):
        result = format_boo2_with_base_prompt("", _tired_state())
        assert len(result) == 1
