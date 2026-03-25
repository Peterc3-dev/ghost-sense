"""Tests for probe selection logic and cooldown enforcement."""

import time

import pytest

from ghost_sense.harvester.prompts import PROBE_TEMPLATES, ProbeSelector
from ghost_sense.models import FieldState, SignalType, StateVector


def _make_state(**fields) -> StateVector:
    sv = StateVector()
    for dim, (val, conf, last_updated) in fields.items():
        sig_type = dim.split(".")[0]
        sv.set(FieldState(
            dimension=dim,
            signal_type=SignalType(sig_type),
            value=val,
            confidence=conf,
            last_updated=last_updated,
        ))
    return sv


@pytest.fixture
def selector():
    return ProbeSelector()


class TestProbeSelection:
    def test_no_probes_when_state_is_fresh(self, selector):
        """High confidence + recent data → no probes needed."""
        now = time.time()
        state = _make_state(**{
            "sleep.quality": (0.8, 0.9, now - 3600),
            "sleep.energy_level": (0.7, 0.85, now - 3600),
            "nutrition.last_meal": (1.0, 0.8, now - 7200),
            "nutrition.quality": (0.7, 0.75, now - 7200),
            "stress.level": (0.3, 0.8, now - 7200),
        })
        probes = selector.select_probes(state, now=now)
        assert len(probes) == 0

    def test_probes_when_sleep_stale(self, selector):
        """Low confidence + old data for sleep → sleep probes fire."""
        now = time.time()
        state = _make_state(**{
            "sleep.quality": (0.5, 0.2, now - 48 * 3600),  # 48h ago, low confidence
            "nutrition.last_meal": (1.0, 0.8, now - 3600),
        })
        probes = selector.select_probes(state, now=now)
        assert len(probes) > 0
        assert any("sleep" in p.template.target_dimension for p in probes)

    def test_probes_when_nutrition_stale(self, selector):
        """Low confidence + old data for nutrition → nutrition probes fire."""
        now = time.time()
        state = _make_state(**{
            "sleep.quality": (0.8, 0.9, now - 3600),
            "nutrition.last_meal": (0.5, 0.15, now - 40 * 3600),  # 40h ago
        })
        probes = selector.select_probes(state, now=now)
        assert len(probes) > 0
        assert any("nutrition" in p.template.target_dimension for p in probes)

    def test_probes_when_dimension_missing(self, selector):
        """No data at all for a dimension → probes fire."""
        now = time.time()
        state = StateVector()  # empty state
        probes = selector.select_probes(state, now=now)
        assert len(probes) > 0

    def test_max_two_probes(self, selector):
        """Never return more than 2 probes."""
        now = time.time()
        state = StateVector()  # everything unknown → many triggers
        probes = selector.select_probes(state, now=now, max_probes=2)
        assert len(probes) <= 2

    def test_priority_ordering(self, selector):
        """Probes should be sorted by priority (ascending)."""
        now = time.time()
        state = StateVector()
        probes = selector.select_probes(state, now=now, max_probes=10)
        for i in range(len(probes) - 1):
            assert probes[i].template.priority <= probes[i + 1].template.priority

    def test_probe_has_reason(self, selector):
        now = time.time()
        state = StateVector()
        probes = selector.select_probes(state, now=now)
        for p in probes:
            assert len(p.reason) > 0


class TestCooldownEnforcement:
    def test_cooldown_blocks_reuse(self, selector):
        """After using a probe, it should be blocked for its cooldown period."""
        now = time.time()
        state = StateVector()  # empty → triggers fire

        probes1 = selector.select_probes(state, now=now)
        assert len(probes1) > 0

        # Record use of all returned probes
        for p in probes1:
            selector.record_use(p.template.id, timestamp=now)

        # Immediately after — same probes should be blocked
        probes2 = selector.select_probes(state, now=now + 60)
        used_ids = {p.template.id for p in probes1}
        for p in probes2:
            assert p.template.id not in used_ids

    def test_cooldown_expires(self, selector):
        """After cooldown period passes, probe becomes available again."""
        now = time.time()
        state = StateVector()

        probes = selector.select_probes(state, now=now)
        assert len(probes) > 0

        first_probe = probes[0]
        selector.record_use(first_probe.template.id, timestamp=now)

        # Jump past cooldown
        future = now + first_probe.template.cooldown_hours * 3600 + 1
        probes_after = selector.select_probes(state, now=future, max_probes=10)
        ids_after = {p.template.id for p in probes_after}
        assert first_probe.template.id in ids_after

    def test_cooldown_just_before_expiry(self, selector):
        """Just before cooldown expires, probe should still be blocked."""
        now = time.time()
        state = StateVector()

        probes = selector.select_probes(state, now=now)
        first_probe = probes[0]
        selector.record_use(first_probe.template.id, timestamp=now)

        # 1 minute before cooldown expires
        almost = now + first_probe.template.cooldown_hours * 3600 - 60
        probes_before = selector.select_probes(state, now=almost, max_probes=10)
        ids_before = {p.template.id for p in probes_before}
        assert first_probe.template.id not in ids_before

    def test_different_probes_independent_cooldowns(self, selector):
        """Cooldown on one probe doesn't affect others."""
        now = time.time()
        state = StateVector()

        probes = selector.select_probes(state, now=now, max_probes=10)
        if len(probes) < 2:
            pytest.skip("Need at least 2 probes for this test")

        # Cool down only the first
        selector.record_use(probes[0].template.id, timestamp=now)

        probes2 = selector.select_probes(state, now=now + 60, max_probes=10)
        ids2 = {p.template.id for p in probes2}
        assert probes[0].template.id not in ids2
        # At least one other probe should still be available
        assert len(probes2) > 0


class TestProbeTextQuality:
    """Probes must be conversational, never clinical."""

    FORBIDDEN = [
        "how did you sleep",
        "how are you feeling",
        "are you okay",
        "have you eaten",
        "are you stressed",
        "mental health",
        "wellbeing",
        "self-care",
        "assessment",
    ]

    def test_no_clinical_language(self):
        for template in PROBE_TEMPLATES:
            lower = template.probe_text.lower()
            for phrase in self.FORBIDDEN:
                assert phrase not in lower, (
                    f"Probe '{template.id}' contains clinical language: '{phrase}'"
                )

    def test_probes_are_short(self):
        """Probes should be brief — conversational, not paragraph-length."""
        for template in PROBE_TEMPLATES:
            word_count = len(template.probe_text.split())
            assert word_count <= 15, (
                f"Probe '{template.id}' too long ({word_count} words): {template.probe_text}"
            )

    def test_all_probes_have_unique_ids(self):
        ids = [t.id for t in PROBE_TEMPLATES]
        assert len(ids) == len(set(ids))
