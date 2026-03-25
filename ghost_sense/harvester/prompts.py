"""Naturalistic probe question templates keyed to state gaps.

Each probe has:
- trigger condition: which dimension, confidence below threshold, hours since last signal
- probe text: conversational, adjacent, never clinical
- cooldown: minimum hours between uses of this probe

These are SUGGESTED injections the conditioner can include — never forced.
The select_probes() function returns 0–2 candidates max, respecting cooldowns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ghost_sense.config import CONFIDENCE_THRESHOLD
from ghost_sense.models import StateVector


@dataclass
class ProbeTemplate:
    """A naturalistic question template tied to a state gap."""

    id: str
    target_dimension: str  # the dimension we're trying to gather data for
    probe_text: str  # the suggested conversational injection
    confidence_below: float  # trigger when confidence drops below this
    hours_since_min: float  # trigger when hours since last signal exceeds this
    cooldown_hours: float  # minimum hours between uses
    priority: int = 5  # lower = higher priority (1–10)


@dataclass
class ProbeCandidate:
    """A probe that has passed all trigger conditions and is ready to inject."""

    template: ProbeTemplate
    reason: str  # human-readable reason this probe was selected


# --- Template bank ---

PROBE_TEMPLATES: list[ProbeTemplate] = [
    # Sleep probes — never ask "how did you sleep" directly
    ProbeTemplate(
        id="sleep_shift_time",
        target_dimension="sleep.quality",
        probe_text="what time you get in tonight?",
        confidence_below=0.3,
        hours_since_min=36,
        cooldown_hours=24,
        priority=3,
    ),
    ProbeTemplate(
        id="sleep_evening_plans",
        target_dimension="sleep.quality",
        probe_text="you staying up late or crashing early?",
        confidence_below=0.35,
        hours_since_min=30,
        cooldown_hours=20,
        priority=4,
    ),
    ProbeTemplate(
        id="sleep_energy_check",
        target_dimension="sleep.energy_level",
        probe_text="you running on fumes or actually good right now?",
        confidence_below=0.25,
        hours_since_min=24,
        cooldown_hours=18,
        priority=4,
    ),

    # Nutrition probes — mention food in passing, see if it draws a response
    ProbeTemplate(
        id="nutrition_food_adjacent",
        target_dimension="nutrition.last_meal",
        probe_text="i could go for some ramen right about now",
        confidence_below=0.3,
        hours_since_min=30,
        cooldown_hours=24,
        priority=4,
    ),
    ProbeTemplate(
        id="nutrition_meal_timing",
        target_dimension="nutrition.last_meal",
        probe_text="you eat yet or still locked in?",
        confidence_below=0.25,
        hours_since_min=36,
        cooldown_hours=20,
        priority=3,
    ),
    ProbeTemplate(
        id="nutrition_cooking",
        target_dimension="nutrition.quality",
        probe_text="you been cooking lately or just surviving?",
        confidence_below=0.3,
        hours_since_min=48,
        cooldown_hours=48,
        priority=6,
    ),

    # Stress probes — ask about workload/schedule, not feelings
    ProbeTemplate(
        id="stress_workload",
        target_dimension="stress.level",
        probe_text="how's the workload looking this week?",
        confidence_below=0.3,
        hours_since_min=48,
        cooldown_hours=48,
        priority=5,
    ),
    ProbeTemplate(
        id="stress_schedule",
        target_dimension="stress.level",
        probe_text="anything big coming up deadline-wise?",
        confidence_below=0.35,
        hours_since_min=36,
        cooldown_hours=36,
        priority=5,
    ),
]


class ProbeSelector:
    """Selects probes based on current state, respecting cooldowns.

    Stateful: tracks when each probe was last used.
    """

    def __init__(self) -> None:
        self._last_used: dict[str, float] = {}  # probe_id -> timestamp

    def record_use(self, probe_id: str, timestamp: float | None = None) -> None:
        """Record that a probe was used at the given time."""
        self._last_used[probe_id] = timestamp or time.time()

    def _is_on_cooldown(self, template: ProbeTemplate, now: float) -> bool:
        last = self._last_used.get(template.id)
        if last is None:
            return False
        hours_since = (now - last) / 3600
        return hours_since < template.cooldown_hours

    def _check_trigger(self, template: ProbeTemplate, state: StateVector, now: float) -> str | None:
        """Check if a probe's trigger conditions are met. Returns reason string or None."""
        fs = state.get(template.target_dimension)

        # Confidence check: trigger if confidence is below threshold OR field is missing
        if fs is not None and fs.confidence >= template.confidence_below:
            return None

        # Hours-since check: trigger if enough time has passed since last signal
        if fs is not None:
            hours_since = (now - fs.last_updated) / 3600
            if hours_since < template.hours_since_min:
                return None
            reason = f"{template.target_dimension} conf={fs.confidence:.2f} (<{template.confidence_below}), {hours_since:.0f}h since last"
        else:
            reason = f"{template.target_dimension} unknown (no data)"

        return reason

    def select_probes(
        self,
        state: StateVector,
        now: float | None = None,
        max_probes: int = 2,
    ) -> list[ProbeCandidate]:
        """Select 0–max_probes candidate probes based on current state.

        Returns candidates sorted by priority (lower number = higher priority).
        Respects cooldowns and trigger conditions.
        """
        now = now or time.time()
        candidates: list[ProbeCandidate] = []

        for template in PROBE_TEMPLATES:
            # Skip if on cooldown
            if self._is_on_cooldown(template, now):
                continue

            # Check trigger conditions
            reason = self._check_trigger(template, state, now)
            if reason is None:
                continue

            candidates.append(ProbeCandidate(template=template, reason=reason))

        # Sort by priority (ascending = higher priority first)
        candidates.sort(key=lambda c: c.template.priority)

        return candidates[:max_probes]
