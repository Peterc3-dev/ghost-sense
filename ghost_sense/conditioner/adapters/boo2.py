"""Adapter: formats preamble for Kimi 2.5 via OpenClaw (OpenAI-compatible API)."""

from __future__ import annotations

from ghost_sense.accumulator.trend import Trend
from ghost_sense.conditioner.preamble import generate_preamble
from ghost_sense.harvester.prompts import ProbeCandidate
from ghost_sense.models import StateVector


def format_boo2_system_message(
    state: StateVector,
    trends: dict[str, dict[int, Trend]] | None = None,
    local_time: str | None = None,
    probes: list[ProbeCandidate] | None = None,
) -> dict:
    """Return the preamble as an OpenAI-format system message for Kimi/OpenClaw."""
    preamble = generate_preamble(state, trends=trends, local_time=local_time, probes=probes)
    return {
        "role": "system",
        "content": preamble,
    }


def format_boo2_with_base_prompt(
    base_system_prompt: str,
    state: StateVector,
    trends: dict[str, dict[int, Trend]] | None = None,
    local_time: str | None = None,
    probes: list[ProbeCandidate] | None = None,
) -> list[dict]:
    """Return system messages list with preamble followed by base prompt."""
    preamble = generate_preamble(state, trends=trends, local_time=local_time, probes=probes)
    messages = [{"role": "system", "content": preamble}]
    if base_system_prompt.strip():
        messages.append({"role": "system", "content": base_system_prompt})
    return messages
