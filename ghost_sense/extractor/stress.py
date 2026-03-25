"""Stress signal extractor — stressor mentions, emotional language, professional pressure."""

from __future__ import annotations

import re
import time

from ghost_sense.extractor.base import BaseExtractor
from ghost_sense.models import SignalEvent, SignalType

# Direct stress/overwhelm
STRESS_PATTERNS = [
    re.compile(r"\b(?:stressed|stressing|stressful|stressed\s+out|so\s+much\s+stress)\b", re.I),
    re.compile(r"\b(?:overwhelmed|overwhelming|drowning\s+in|buried\s+in|swamped)\b", re.I),
    re.compile(r"\b(?:anxious|anxiety|panicking|freaking\s+out|losing\s+(?:it|my\s+mind))\b", re.I),
    re.compile(r"\b(?:can'?t\s+(?:handle|deal|cope|take\s+it|breathe))\b", re.I),
    re.compile(r"\b(?:breaking\s+(?:point|down)|meltdown|falling\s+apart|losing\s+my\s+shit)\b", re.I),
    re.compile(r"\b(?:too\s+much|way\s+too\s+much|so\s+much\s+(?:to\s+do|going\s+on|pressure))\b", re.I),
]

# Professional pressure
WORK_PRESSURE_PATTERNS = [
    re.compile(r"\b(?:deadline|deadlines|due\s+(?:date|today|tomorrow|soon))\b", re.I),
    re.compile(r"\b(?:boss|manager|client)\s+(?:is|wants|needs|keeps|won'?t\s+stop)\b", re.I),
    re.compile(r"\b(?:behind\s+(?:on|schedule)|running\s+late|overdue|past\s+due)\b", re.I),
    re.compile(r"\b(?:crunch(?:ing|time)?|on\s+call|pager|incident|outage|prod(?:uction)?\s+(?:down|issue|fire))\b", re.I),
    re.compile(r"\b(?:sprint\s+(?:ending|review)|demo\s+(?:tomorrow|today|soon))\b", re.I),
    re.compile(r"\b(?:overtime|working\s+(?:late|weekends?|overtime)|double\s+shift)\b", re.I),
]

# Emotional distress (non-professional)
EMOTIONAL_PATTERNS = [
    re.compile(r"\b(?:depressed|depression|hopeless|worthless|empty(?:\s+inside)?)\b", re.I),
    re.compile(r"\b(?:frustrated|frustrating|pissed|furious|livid|rage)\b", re.I),
    re.compile(r"\b(?:burned?\s*out|burnout|burnt?\s*out)\b", re.I),
    re.compile(r"\b(?:lonely|isolated|alone|nobody\s+(?:cares|gets\s+it))\b", re.I),
    re.compile(r"\b(?:crying|cried|tears|broke\s+down)\b", re.I),
]

# Low stress / calm signals
CALM_PATTERNS = [
    re.compile(r"\b(?:relaxed|relaxing|chill(?:ing)?|peaceful|calm|zen)\b", re.I),
    re.compile(r"\b(?:no\s+(?:stress|worries|pressure)|stress[\s-]?free|easy\s+(?:day|week))\b", re.I),
    re.compile(r"\b(?:caught\s+up|ahead\s+of\s+schedule|smooth\s+sailing|all\s+good)\b", re.I),
]


def _match_any(text: str, patterns: list[re.Pattern]) -> list[str]:
    hits = []
    for p in patterns:
        m = p.search(text)
        if m:
            hits.append(m.group())
    return hits


class StressExtractor(BaseExtractor):
    """Extracts stress level and stress source signals."""

    def extract(self, message: str, metadata: dict) -> list[SignalEvent]:
        ts = metadata.get("timestamp", time.time())
        events: list[SignalEvent] = []

        stress_hits = _match_any(message, STRESS_PATTERNS)
        work_hits = _match_any(message, WORK_PRESSURE_PATTERNS)
        emotional_hits = _match_any(message, EMOTIONAL_PATTERNS)
        calm_hits = _match_any(message, CALM_PATTERNS)

        # --- Stress level ---
        # Aggregate: more pattern matches = higher confidence
        high_stress_count = len(stress_hits) + len(work_hits) + len(emotional_hits)
        low_stress_count = len(calm_hits)

        if high_stress_count > 0 and high_stress_count > low_stress_count:
            # Scale: 1 match = moderate (0.6), 2+ = high (0.8+)
            level = min(0.5 + high_stress_count * 0.15, 1.0)
            confidence = min(0.6 + high_stress_count * 0.1, 0.95)
            source = (stress_hits + work_hits + emotional_hits)[0]
            events.append(SignalEvent(
                signal_type=SignalType.STRESS,
                dimension="stress.level",
                value=level,
                confidence=confidence,
                source_text=source,
                timestamp=ts,
            ))
        elif low_stress_count > 0 and high_stress_count == 0:
            events.append(SignalEvent(
                signal_type=SignalType.STRESS,
                dimension="stress.level",
                value=0.15,
                confidence=0.65,
                source_text=calm_hits[0],
                timestamp=ts,
            ))

        # --- Stress source ---
        if work_hits:
            events.append(SignalEvent(
                signal_type=SignalType.STRESS,
                dimension="stress.source",
                value=1.0,  # 1.0 = professional
                confidence=0.75,
                source_text=work_hits[0],
                timestamp=ts,
                metadata={"source_type": "professional"},
            ))
        elif emotional_hits and not work_hits:
            events.append(SignalEvent(
                signal_type=SignalType.STRESS,
                dimension="stress.source",
                value=0.0,  # 0.0 = personal/emotional
                confidence=0.7,
                source_text=emotional_hits[0],
                timestamp=ts,
                metadata={"source_type": "personal"},
            ))

        return events
