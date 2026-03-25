"""Tests for CIN routing hint generation."""

import time

import pytest

from ghost_sense.models import FieldState, SignalType, StateVector
from ghost_sense.routing.cin_feedback import (
    CINRoutingHint,
    PriorityContext,
    ProbingLevel,
    generate_routing_hint,
)


def _make_state(**fields) -> StateVector:
    sv = StateVector()
    now = time.time()
    for dim, val in fields.items():
        sig_type = dim.split(".")[0]
        sv.set(FieldState(
            dimension=dim,
            signal_type=SignalType(sig_type),
            value=val,
            confidence=0.85,
            last_updated=now,
        ))
    return sv


class TestFatigueRouting:
    def test_low_energy_increases_delegation(self):
        state = _make_state(**{"sleep.energy_level": 0.2})
        hint = generate_routing_hint(state)
        assert hint.delegation_weight > 0.5
        assert any("energy" in r.lower() or "delegation" in r.lower() for r in hint.reasons)

    def test_sleep_deficit_increases_delegation(self):
        state = _make_state(**{"sleep.quality": 0.2})
        hint = generate_routing_hint(state)
        assert hint.delegation_weight > 0.4
        assert hint.priority_context == PriorityContext.LOW_COGNITIVE

    def test_combined_fatigue_high_delegation(self):
        state = _make_state(**{
            "sleep.energy_level": 0.15,
            "sleep.quality": 0.2,
        })
        hint = generate_routing_hint(state)
        assert hint.delegation_weight > 0.7
        assert hint.probing_level == ProbingLevel.MINIMAL

    def test_good_sleep_no_fatigue_boost(self):
        state = _make_state(**{
            "sleep.energy_level": 0.8,
            "sleep.quality": 0.9,
        })
        hint = generate_routing_hint(state)
        assert hint.delegation_weight < 0.5


class TestExecutionMode:
    def test_high_energy_formal_execution_mode(self):
        state = _make_state(**{
            "sleep.energy_level": 0.9,
            "register.formality_score": 0.8,
        })
        hint = generate_routing_hint(state)
        assert hint.priority_context == PriorityContext.EXECUTION
        assert hint.probing_level == ProbingLevel.NONE
        assert any("execution" in r.lower() for r in hint.reasons)

    def test_high_energy_casual_not_execution(self):
        state = _make_state(**{
            "sleep.energy_level": 0.9,
            "register.formality_score": 0.3,
        })
        hint = generate_routing_hint(state)
        assert hint.priority_context != PriorityContext.EXECUTION


class TestStressRouting:
    def test_professional_stress_task_support(self):
        state = _make_state(**{
            "stress.level": 0.8,
            "stress.source": 1.0,  # professional
        })
        hint = generate_routing_hint(state)
        assert hint.priority_context == PriorityContext.TASK_SUPPORT
        assert hint.probing_level == ProbingLevel.MINIMAL
        assert any("task support" in r.lower() for r in hint.reasons)

    def test_personal_stress_lighter_touch(self):
        state = _make_state(**{
            "stress.level": 0.8,
            "stress.source": 0.0,  # personal
        })
        hint = generate_routing_hint(state)
        assert any("personal" in r.lower() or "lighter" in r.lower() for r in hint.reasons)

    def test_low_stress_no_special_routing(self):
        state = _make_state(**{"stress.level": 0.2})
        hint = generate_routing_hint(state)
        assert hint.priority_context != PriorityContext.TASK_SUPPORT


class TestCadenceRouting:
    def test_burst_reduces_delegation(self):
        state = _make_state(**{"cadence.burst_score": 0.9})
        hint = generate_routing_hint(state)
        # Delegation should be low — user is rapid-fire, minimize latency
        assert hint.delegation_weight < 0.3
        assert any("rapid" in r.lower() or "cadence" in r.lower() for r in hint.reasons)


class TestLowConfidenceProbing:
    def test_empty_state_elevates_probing(self):
        state = StateVector()
        hint = generate_routing_hint(state)
        # No fields → avg confidence 0 → should elevate probing
        # (but only if probing isn't already set to NONE)
        assert hint.probing_level in (ProbingLevel.ELEVATED, ProbingLevel.NORMAL)

    def test_low_confidence_fields_elevate_probing(self):
        now = time.time()
        sv = StateVector()
        # Add fields with very low confidence
        for dim in ["sleep.quality", "nutrition.last_meal", "stress.level"]:
            sig_type = dim.split(".")[0]
            sv.set(FieldState(
                dimension=dim,
                signal_type=SignalType(sig_type),
                value=0.5,
                confidence=0.2,
                last_updated=now,
            ))
        hint = generate_routing_hint(sv)
        assert hint.probing_level == ProbingLevel.ELEVATED


class TestRoutingHintStructure:
    def test_delegation_clamped(self):
        """Delegation weight must be in [0.0, 1.0]."""
        # Extreme fatigue state
        state = _make_state(**{
            "sleep.energy_level": 0.05,
            "sleep.quality": 0.05,
            "stress.level": 0.95,
            "stress.source": 0.0,
        })
        hint = generate_routing_hint(state)
        assert 0.0 <= hint.delegation_weight <= 1.0

    def test_always_has_reasons(self):
        hint = generate_routing_hint(StateVector())
        assert len(hint.reasons) > 0

    def test_baseline_on_empty_state(self):
        hint = generate_routing_hint(StateVector())
        assert isinstance(hint, CINRoutingHint)
        assert isinstance(hint.probing_level, ProbingLevel)
        assert isinstance(hint.priority_context, PriorityContext)


class TestCompositeScenarios:
    def test_tired_stressed_professional(self):
        """Tired + professional stress → high delegation + task support."""
        state = _make_state(**{
            "sleep.energy_level": 0.2,
            "sleep.quality": 0.3,
            "stress.level": 0.8,
            "stress.source": 1.0,
        })
        hint = generate_routing_hint(state)
        assert hint.delegation_weight > 0.5
        assert hint.priority_context == PriorityContext.TASK_SUPPORT

    def test_energized_casual_burst(self):
        """High energy + casual + burst → low delegation, responsive."""
        state = _make_state(**{
            "sleep.energy_level": 0.9,
            "register.formality_score": 0.3,
            "cadence.burst_score": 0.9,
        })
        hint = generate_routing_hint(state)
        assert hint.delegation_weight < 0.3
