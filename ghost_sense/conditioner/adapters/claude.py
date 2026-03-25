"""Adapter: formats preamble for Claude API / Claude Code system prompts."""

from __future__ import annotations

from ghost_sense.accumulator.trend import Trend
from ghost_sense.conditioner.preamble import generate_preamble
from ghost_sense.harvester.prompts import ProbeCandidate
from ghost_sense.models import StateVector


def format_claude_system_prompt(
    base_system_prompt: str,
    state: StateVector,
    trends: dict[str, dict[int, Trend]] | None = None,
    local_time: str | None = None,
    probes: list[ProbeCandidate] | None = None,
) -> str:
    """Prepend ambient context preamble to a Claude system prompt."""
    preamble = generate_preamble(state, trends=trends, local_time=local_time, probes=probes)

    if not base_system_prompt.strip():
        return preamble

    return f"{preamble}\n\n{base_system_prompt}"


def format_claude_api_messages(
    state: StateVector,
    trends: dict[str, dict[int, Trend]] | None = None,
    local_time: str | None = None,
    probes: list[ProbeCandidate] | None = None,
) -> dict:
    """Return the preamble as a system message dict for the Claude API."""
    preamble = generate_preamble(state, trends=trends, local_time=local_time, probes=probes)
    return {
        "type": "text",
        "text": preamble,
        "cache_control": {"type": "ephemeral"},
    }
