"""Sleep/wake/fatigue signal extractor — keyword and pattern matching."""

from __future__ import annotations

import re
import time

from ghost_sense.extractor.base import BaseExtractor
from ghost_sense.models import SignalEvent, SignalType

# Explicit sleep mentions with quality/duration hints
POOR_SLEEP_PATTERNS = [
    re.compile(r"\b(?:barely|hardly|didn'?t|couldn'?t|can'?t)\s+(?:slept?|sleep)\b", re.I),
    re.compile(r"\b(?:got|had|only)\s+(?:\d{1}|few|couple)\s*(?:hours?|hrs?)\s*(?:of\s+)?sleep\b", re.I),
    re.compile(r"\b(?:insomnia|sleepless|restless\s+night|tossed?\s+and\s+turned?)\b", re.I),
    re.compile(r"\bwoke\s+up\s+(?:like\s+)?(?:\d+|a\s+bunch|multiple|several)\s+times?\b", re.I),
    re.compile(r"\b(?:been\s+up\s+(?:all\s+night|since|for\s+\d+))\b", re.I),
    re.compile(r"\b(?:pulled\s+an?\s+all[\s-]?nighter)\b", re.I),
    re.compile(r"\b(?:no\s+sleep|zero\s+sleep|0\s+sleep)\b", re.I),
]

GOOD_SLEEP_PATTERNS = [
    re.compile(r"\b(?:slept?\s+(?:well|great|good|solid|like\s+a\s+(?:rock|log|baby)))\b", re.I),
    re.compile(r"\b(?:got|had)\s+(?:a\s+)?(?:good|great|solid|full|8|9|10)\s*(?:hours?|hrs?)?\s*(?:of\s+)?sleep\b", re.I),
    re.compile(r"\b(?:well[\s-]?rested|refreshed|recharged)\b", re.I),
    re.compile(r"\b(?:passed\s+out|knocked\s+out|out\s+like\s+a\s+light)\b", re.I),
]

# Caffeine as sleep proxy
CAFFEINE_PATTERNS = [
    re.compile(r"\b(?:coffee|espresso|caffeine|red\s*bull|energy\s*drink|monster|bang)\b", re.I),
    re.compile(r"\b(?:second|third|fourth|2nd|3rd|4th|\d+(?:th|rd|nd))\s+(?:coffee|cup|espresso)\b", re.I),
]

LATE_CAFFEINE_PATTERN = re.compile(
    r"\b(?:coffee|espresso|caffeine|energy\s*drink)\s+(?:at\s+)?(?:1[0-9]|2[0-3]|midnight|late|tonight)\b", re.I
)

# Energy/fatigue language
FATIGUE_PATTERNS = [
    re.compile(r"\b(?:exhausted|wiped|drained|dead\s+tired|running\s+on\s+(?:empty|fumes))\b", re.I),
    re.compile(r"\b(?:so\s+tired|super\s+tired|really\s+tired|extremely\s+tired|tired\s+af)\b", re.I),
    re.compile(r"\b(?:can'?t\s+(?:keep\s+(?:my\s+)?eyes\s+open|stay\s+awake|focus|think))\b", re.I),
    re.compile(r"\b(?:brain\s+(?:fog|dead|fried)|fried|zonked|zombie)\b", re.I),
    re.compile(r"\b(?:need\s+(?:a\s+)?(?:nap|sleep|rest|coffee))\b", re.I),
    re.compile(r"\b(?:falling\s+asleep|dozing\s+off|nodding\s+off)\b", re.I),
]

ENERGY_PATTERNS = [
    re.compile(r"\b(?:wired|buzzing|energized|pumped|hyped|amped|locked\s+in)\b", re.I),
    re.compile(r"\b(?:wide\s+awake|fully\s+awake|feeling\s+(?:great|good|sharp|alert))\b", re.I),
]


def _match_any(text: str, patterns: list[re.Pattern]) -> list[str]:
    """Return all matches from a pattern list."""
    hits = []
    for p in patterns:
        m = p.search(text)
        if m:
            hits.append(m.group())
    return hits


class SleepExtractor(BaseExtractor):
    """Extracts sleep quality, caffeine proxy, and energy level signals."""

    def extract(self, message: str, metadata: dict) -> list[SignalEvent]:
        ts = metadata.get("timestamp", time.time())
        local_time = metadata.get("local_time", "12:00")
        events: list[SignalEvent] = []

        # --- Sleep quality ---
        poor_hits = _match_any(message, POOR_SLEEP_PATTERNS)
        good_hits = _match_any(message, GOOD_SLEEP_PATTERNS)

        if poor_hits:
            events.append(SignalEvent(
                signal_type=SignalType.SLEEP,
                dimension="sleep.quality",
                value=0.2,  # 0=terrible, 1=great
                confidence=0.85,
                source_text=poor_hits[0],
                timestamp=ts,
            ))
        elif good_hits:
            events.append(SignalEvent(
                signal_type=SignalType.SLEEP,
                dimension="sleep.quality",
                value=0.9,
                confidence=0.85,
                source_text=good_hits[0],
                timestamp=ts,
            ))

        # --- Caffeine proxy ---
        caffeine_hits = _match_any(message, CAFFEINE_PATTERNS)
        if caffeine_hits:
            # Late caffeine is a stronger sleep deficit signal
            late_hit = LATE_CAFFEINE_PATTERN.search(message)
            is_late = late_hit is not None

            # Also check local_time for late hours
            try:
                hour = int(local_time.split(":")[0])
                is_late = is_late or hour >= 21 or hour < 4
            except (ValueError, IndexError):
                pass

            value = 0.3 if is_late else 0.6  # lower = worse sleep proxy
            events.append(SignalEvent(
                signal_type=SignalType.SLEEP,
                dimension="sleep.caffeine_proxy",
                value=value,
                confidence=0.6 if not is_late else 0.75,
                source_text=caffeine_hits[0],
                timestamp=ts,
                metadata={"late_caffeine": is_late},
            ))

        # --- Energy level ---
        fatigue_hits = _match_any(message, FATIGUE_PATTERNS)
        energy_hits = _match_any(message, ENERGY_PATTERNS)

        if fatigue_hits:
            events.append(SignalEvent(
                signal_type=SignalType.SLEEP,
                dimension="sleep.energy_level",
                value=0.2,
                confidence=0.8,
                source_text=fatigue_hits[0],
                timestamp=ts,
            ))
        elif energy_hits:
            events.append(SignalEvent(
                signal_type=SignalType.SLEEP,
                dimension="sleep.energy_level",
                value=0.9,
                confidence=0.75,
                source_text=energy_hits[0],
                timestamp=ts,
            ))

        return events
