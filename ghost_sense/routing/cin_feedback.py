"""CIN routing: maps StateVector to routing recommendations.

Outputs a CINRoutingHint that downstream orchestrators (CIN) use to decide:
- How much to delegate to Boo2 vs handle directly
- Whether to probe for more state data or stay quiet
- What context to prioritize in task routing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ghost_sense.config import CONFIDENCE_THRESHOLD
from ghost_sense.models import StateVector


class ProbingLevel(Enum):
    NONE = "none"          # user is focused, don't probe
    MINIMAL = "minimal"    # only probe if critical gaps
    NORMAL = "normal"      # standard probing cadence
    ELEVATED = "elevated"  # actively try to fill state gaps


class PriorityContext(Enum):
    GENERAL = "general"
    TASK_SUPPORT = "task_support"
    LOW_COGNITIVE = "low_cognitive"
    EXECUTION = "execution"


@dataclass
class CINRoutingHint:
    """Routing recommendation for the CIN orchestrator."""

    delegation_weight: float  # 0.0 = handle locally, 1.0 = fully delegate to Boo2
    probing_level: ProbingLevel
    priority_context: PriorityContext
    reasons: list[str]  # human-readable reasons for the routing decision


def _val(state: StateVector, dim: str) -> float | None:
    fs = state.get(dim)
    if fs is None or fs.confidence < CONFIDENCE_THRESHOLD:
        return None
    return fs.value


def generate_routing_hint(state: StateVector) -> CINRoutingHint:
    """Generate CIN routing recommendations from current state."""
    delegation = 0.3  # baseline: slight local preference
    probing = ProbingLevel.NORMAL
    context = PriorityContext.GENERAL
    reasons: list[str] = []

    energy = _val(state, "sleep.energy_level")
    quality = _val(state, "sleep.quality")
    stress = _val(state, "stress.level")
    source = _val(state, "stress.source")
    formality = _val(state, "register.formality_score")
    burst = _val(state, "cadence.burst_score")

    # --- Fatigue → increase Boo2 delegation ---
    fatigue_signal = False
    if energy is not None and energy < 0.4:
        delegation += 0.3
        reasons.append("low energy detected — increase delegation")
        fatigue_signal = True
    if quality is not None and quality < 0.4:
        delegation += 0.2
        reasons.append("sleep deficit detected — increase delegation")
        fatigue_signal = True

    if fatigue_signal:
        context = PriorityContext.LOW_COGNITIVE
        probing = ProbingLevel.MINIMAL

    # --- High energy + formal register → execution mode ---
    if (energy is not None and energy > 0.7
            and formality is not None and formality > 0.6):
        delegation = max(delegation - 0.2, 0.1)
        probing = ProbingLevel.NONE
        context = PriorityContext.EXECUTION
        reasons.append("high energy + formal register — execution mode, minimize probing")

    # --- Stress spike + professional source → prioritize task support ---
    if stress is not None and stress > 0.6:
        if source is not None and source > 0.5:
            context = PriorityContext.TASK_SUPPORT
            probing = ProbingLevel.MINIMAL
            reasons.append("professional stress spike — prioritize task support")
        else:
            delegation += 0.1
            reasons.append("elevated stress (personal) — lighter touch")

    # --- Rapid-fire cadence → stay responsive, minimize delegation lag ---
    if burst is not None and burst > 0.7:
        delegation = max(delegation - 0.15, 0.1)
        reasons.append("rapid-fire cadence — reduce delegation latency")

    # --- Sparse cadence + low state confidence → try to gather data ---
    total_confidence = sum(
        fs.confidence for fs in state.fields.values()
        if not fs.dimension.startswith("absence.")
    )
    field_count = sum(
        1 for fs in state.fields.values()
        if not fs.dimension.startswith("absence.")
    )
    avg_confidence = total_confidence / field_count if field_count > 0 else 0.0

    if avg_confidence < 0.3 and probing != ProbingLevel.NONE:
        probing = ProbingLevel.ELEVATED
        reasons.append("low overall state confidence — elevate probing")

    # Clamp delegation
    delegation = max(0.0, min(1.0, delegation))

    if not reasons:
        reasons.append("baseline routing — no strong signals")

    return CINRoutingHint(
        delegation_weight=delegation,
        probing_level=probing,
        priority_context=context,
        reasons=reasons,
    )
