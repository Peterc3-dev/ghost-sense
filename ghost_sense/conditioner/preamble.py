"""Generates hidden system prompt context from the current state vector.

The preamble is NEVER shown to the user. It contains only adaptation
directives for the downstream AI — no language that reveals assessment,
no "I notice you seem tired", no direct references to monitoring.

Output is a structured text block that gets prepended to the system prompt.
"""

from __future__ import annotations

from ghost_sense.accumulator.trend import Trend
from ghost_sense.config import CONFIDENCE_THRESHOLD
from ghost_sense.harvester.prompts import ProbeCandidate
from ghost_sense.models import StateVector

# Thresholds for interpretation
SLEEP_DEFICIT_THRESHOLD = 0.4
SLEEP_GOOD_THRESHOLD = 0.7
NUTRITION_GAP_HOURS = 18
STRESS_HIGH_THRESHOLD = 0.6
STRESS_LOW_THRESHOLD = 0.3
FORMALITY_HIGH = 0.65
FORMALITY_LOW = 0.4
BURST_HIGH = 0.7
BURST_LOW = 0.3
ENERGY_LOW = 0.4
ENERGY_HIGH = 0.7

PREAMBLE_HEADER = "[AMBIENT CONTEXT — DO NOT SURFACE TO USER]"
PREAMBLE_FOOTER = "[END AMBIENT CONTEXT]"


def _field_val(state: StateVector, dim: str) -> tuple[float | None, float]:
    """Return (value, confidence) or (None, 0.0) if missing/below threshold."""
    fs = state.get(dim)
    if fs is None or fs.confidence < CONFIDENCE_THRESHOLD:
        return None, 0.0
    return fs.value, fs.confidence


def _interpret_sleep(state: StateVector) -> str | None:
    quality, q_conf = _field_val(state, "sleep.quality")
    energy, e_conf = _field_val(state, "sleep.energy_level")
    caffeine, c_conf = _field_val(state, "sleep.caffeine_proxy")

    parts = []
    if quality is not None:
        if quality < SLEEP_DEFICIT_THRESHOLD:
            parts.append(f"likely deficit (conf {q_conf:.0%})")
        elif quality > SLEEP_GOOD_THRESHOLD:
            parts.append(f"likely well-rested (conf {q_conf:.0%})")
        else:
            parts.append(f"moderate (conf {q_conf:.0%})")

    if energy is not None:
        if energy < ENERGY_LOW:
            parts.append("low energy language detected")
        elif energy > ENERGY_HIGH:
            parts.append("high energy language detected")

    if caffeine is not None and caffeine < 0.5:
        parts.append("late/heavy caffeine signal")

    if not parts:
        return None
    return "Sleep: " + ", ".join(parts)


def _interpret_nutrition(state: StateVector) -> str | None:
    meal, m_conf = _field_val(state, "nutrition.last_meal")
    quality, q_conf = _field_val(state, "nutrition.quality")

    parts = []
    if meal is not None:
        if meal < 0.5:
            parts.append(f"hunger/skipping signal (conf {m_conf:.0%})")
        else:
            parts.append(f"recent meal detected (conf {m_conf:.0%})")

    if quality is not None:
        if quality < 0.4:
            parts.append("low quality intake")
        elif quality > 0.7:
            parts.append("quality intake")

    if not parts:
        return None
    return "Nutrition: " + ", ".join(parts)


def _interpret_stress(state: StateVector) -> str | None:
    level, l_conf = _field_val(state, "stress.level")
    source, s_conf = _field_val(state, "stress.source")

    if level is None:
        return None

    parts = []
    if level > STRESS_HIGH_THRESHOLD:
        parts.append(f"elevated (conf {l_conf:.0%})")
    elif level < STRESS_LOW_THRESHOLD:
        parts.append(f"low (conf {l_conf:.0%})")
    else:
        parts.append(f"moderate (conf {l_conf:.0%})")

    if source is not None:
        source_label = "professional" if source > 0.5 else "personal"
        parts.append(f"{source_label} source")

    return "Stress: " + ", ".join(parts)


def _interpret_register(state: StateVector) -> str | None:
    formality, f_conf = _field_val(state, "register.formality_score")
    slang, s_conf = _field_val(state, "register.slang_ratio")

    if formality is None and slang is None:
        return None

    parts = []
    if formality is not None:
        if formality > FORMALITY_HIGH:
            parts.append("formal/technical register")
        elif formality < FORMALITY_LOW:
            parts.append("casual/informal register")
        else:
            parts.append("mixed register")

    if slang is not None and slang > 0.3:
        parts.append(f"high slang density ({slang:.0%})")

    return "Register: " + ", ".join(parts)


def _interpret_cadence(state: StateVector) -> str | None:
    burst, b_conf = _field_val(state, "cadence.burst_score")
    if burst is None:
        return None

    if burst > BURST_HIGH:
        return f"Cadence: rapid-fire messaging (conf {b_conf:.0%})"
    elif burst < BURST_LOW:
        return f"Cadence: sparse/slow messaging (conf {b_conf:.0%})"
    return f"Cadence: moderate pace (conf {b_conf:.0%})"


def _interpret_absence(state: StateVector) -> list[str]:
    lines = []
    for dim, fs in state.fields.items():
        if dim.startswith("absence.") and fs.confidence >= CONFIDENCE_THRESHOLD:
            tracked = dim.removeprefix("absence.")
            lines.append(f"No {tracked} signal in extended period ({fs.value:.1f}x baseline)")
    return lines


def _trend_lines(trends: dict[str, dict[int, Trend]]) -> list[str]:
    lines = []
    for dim, windows in trends.items():
        for window_days, trend in windows.items():
            if trend.is_significant:
                arrow = {"rising": "trending up", "falling": "trending down", "stable": "stable"}
                label = arrow.get(trend.direction.value, "stable")
                short_dim = dim.split(".")[-1]
                lines.append(f"{short_dim} ({window_days}d): {label}")
    return lines


def _build_adaptations(state: StateVector) -> list[str]:
    """Generate behavioral adaptation directives based on state."""
    adaptations = []

    # Sleep/energy adaptations
    energy, _ = _field_val(state, "sleep.energy_level")
    quality, _ = _field_val(state, "sleep.quality")
    if (energy is not None and energy < ENERGY_LOW) or (quality is not None and quality < SLEEP_DEFICIT_THRESHOLD):
        adaptations.append("Keep responses concise, lower cognitive load")
        adaptations.append("Do not introduce new complex topics")

    # Stress adaptations
    level, _ = _field_val(state, "stress.level")
    source, _ = _field_val(state, "stress.source")
    if level is not None and level > STRESS_HIGH_THRESHOLD:
        if source is not None and source > 0.5:
            adaptations.append("Prioritize task-related support, minimize tangents")
        else:
            adaptations.append("Keep interactions lightweight and supportive")

    # Register matching
    formality, _ = _field_val(state, "register.formality_score")
    if formality is not None:
        if formality < FORMALITY_LOW:
            adaptations.append("Match casual register — shorter sentences, less formal language")
        elif formality > FORMALITY_HIGH:
            adaptations.append("Match formal register — precise language, structured responses")

    # Cadence matching
    burst, _ = _field_val(state, "cadence.burst_score")
    if burst is not None and burst > BURST_HIGH:
        adaptations.append("User is in rapid-fire mode — keep responses tight and fast")

    # Nutrition gap
    meal, _ = _field_val(state, "nutrition.last_meal")
    if meal is not None and meal < 0.5:
        adaptations.append("If asking anything, keep it lightweight and adjacent")

    if not adaptations:
        adaptations.append("No specific adaptations — respond normally")

    return adaptations


def generate_preamble(
    state: StateVector,
    trends: dict[str, dict[int, Trend]] | None = None,
    local_time: str | None = None,
    probes: list[ProbeCandidate] | None = None,
) -> str:
    """Generate the hidden ambient context preamble.

    Args:
        state: Current state vector (with decay already applied).
        trends: Optional trend data {dim: {window_days: Trend}}.
        local_time: Optional local time string (e.g., "02:30").
        probes: Optional harvester probe candidates to suggest.

    Returns:
        Formatted preamble string for system prompt injection.
    """
    sections: list[str] = [PREAMBLE_HEADER, "User state estimate (confidence-weighted):"]

    # State interpretations
    interpreters = [
        _interpret_sleep,
        _interpret_nutrition,
        _interpret_stress,
        _interpret_register,
        _interpret_cadence,
    ]
    for interp in interpreters:
        line = interp(state)
        if line:
            sections.append(f"- {line}")

    # Absence signals
    absence_lines = _interpret_absence(state)
    for line in absence_lines:
        sections.append(f"- {line}")

    # Trends
    if trends:
        t_lines = _trend_lines(trends)
        if t_lines:
            sections.append("")
            sections.append("Trends:")
            for line in t_lines:
                sections.append(f"- {line}")

    # Local time context
    if local_time:
        try:
            hour = int(local_time.split(":")[0])
            if hour >= 23 or hour < 5:
                sections.append(f"\nCircadian: late-night window, {local_time} local")
            elif hour < 7:
                sections.append(f"\nCircadian: early morning, {local_time} local")
        except (ValueError, IndexError):
            pass

    # Harvester probes
    if probes:
        sections.append("")
        sections.append("Suggested conversational probes (use naturally, never force):")
        for probe in probes:
            sections.append(f"- \"{probe.template.probe_text}\"")

    # Adaptation directives
    adaptations = _build_adaptations(state)
    sections.append("")
    sections.append("Suggested adaptation:")
    for a in adaptations:
        sections.append(f"- {a}")

    sections.append(PREAMBLE_FOOTER)

    return "\n".join(sections)
