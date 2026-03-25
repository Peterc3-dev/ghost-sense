"""Core data models for ghost-sense."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class SignalType(Enum):
    SLEEP = "sleep"
    NUTRITION = "nutrition"
    STRESS = "stress"
    REGISTER = "register"
    CADENCE = "cadence"
    ABSENCE = "absence"


@dataclass
class SignalEvent:
    """A single extracted signal from a message."""

    signal_type: SignalType
    dimension: str  # sub-field, e.g. "slang_ratio", "avg_word_length"
    value: float
    confidence: float  # 0.0 – 1.0
    source_text: str  # the triggering snippet
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """A point-in-time snapshot of the full state vector, stored for trend analysis."""

    timestamp: float
    fields: dict[str, FieldState]  # dimension -> FieldState


@dataclass
class FieldState:
    """A single field within the state vector."""

    dimension: str
    signal_type: SignalType
    value: float
    confidence: float
    last_updated: float


@dataclass
class StateVector:
    """The full ambient state estimate for a user."""

    fields: dict[str, FieldState] = field(default_factory=dict)

    def get(self, dimension: str) -> FieldState | None:
        return self.fields.get(dimension)

    def set(self, fs: FieldState) -> None:
        self.fields[fs.dimension] = fs
