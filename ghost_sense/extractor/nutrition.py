"""Nutrition signal extractor — food mentions, meal timing, quality signals."""

from __future__ import annotations

import re
import time

from ghost_sense.extractor.base import BaseExtractor
from ghost_sense.models import SignalEvent, SignalType

# Meal mentions
MEAL_PATTERNS = [
    re.compile(r"\b(?:breakfast|lunch|dinner|supper|brunch|snack(?:ing|ed|s)?)\b", re.I),
    re.compile(r"\b(?:ate|eating|eaten|had\s+(?:some\s+)?(?:food|a\s+meal))\b", re.I),
    re.compile(r"\b(?:just\s+(?:ate|had|grabbed|made)\b)", re.I),
    re.compile(r"\b(?:cooking|cooked|made\s+(?:food|dinner|lunch))\b", re.I),
    re.compile(r"\b(?:ordered\s+(?:food|pizza|takeout|delivery))\b", re.I),
]

# Specific food items (evidence a meal happened)
FOOD_ITEMS = re.compile(
    r"\b(?:pizza|burger|sandwich|salad|pasta|rice|chicken|steak|sushi|ramen|noodles|"
    r"tacos?|burrito|soup|eggs?|toast|cereal|oatmeal|yogurt|fruit|veggies|"
    r"fries|chips|wings|wrap|bowl|curry|stir\s*fry|"
    r"smoothie|protein\s*(?:shake|bar)|granola)\b",
    re.I,
)

# Quality signals
HIGH_QUALITY_PATTERNS = [
    re.compile(r"\b(?:clean\s+meal|home[\s-]?cooked|healthy|fresh|whole\s+foods?|balanced)\b", re.I),
    re.compile(r"\b(?:meal\s+prep|prepped|salad|veggies|vegetables|grilled|steamed|baked)\b", re.I),
    re.compile(r"\b(?:protein|fiber|nutrients|nutritious|wholesome)\b", re.I),
]

LOW_QUALITY_PATTERNS = [
    re.compile(r"\b(?:fast\s+food|junk\s+food|grabbed\s+(?:some|a)\s+(?:quick|fast))\b", re.I),
    re.compile(r"\b(?:mcdonalds?|mcdicks|wendys?|taco\s+bell|kfc|burger\s+king)\b", re.I),
    re.compile(r"\b(?:gas\s+station|vending\s+machine|convenience\s+store)\b", re.I),
    re.compile(r"\b(?:instant\s+(?:noodles|ramen)|cup\s+noodles?|microwave(?:d|\s+meal))\b", re.I),
    re.compile(r"\b(?:trash|garbage|terrible)\s+(?:food|meal|diet)\b", re.I),
]

# Hunger/skipping
HUNGER_PATTERNS = [
    re.compile(r"\b(?:starving|famished|so\s+hungry|haven'?t\s+eaten|forgot\s+to\s+eat)\b", re.I),
    re.compile(r"\b(?:skipped\s+(?:breakfast|lunch|dinner|meals?|eating))\b", re.I),
    re.compile(r"\b(?:no\s+(?:time\s+to\s+)?eat|didn'?t\s+eat|can'?t\s+eat)\b", re.I),
]


def _match_any(text: str, patterns: list[re.Pattern]) -> list[str]:
    hits = []
    for p in patterns:
        m = p.search(text)
        if m:
            hits.append(m.group())
    return hits


class NutritionExtractor(BaseExtractor):
    """Extracts meal occurrence, meal quality, and hunger/skipping signals."""

    def extract(self, message: str, metadata: dict) -> list[SignalEvent]:
        ts = metadata.get("timestamp", time.time())
        events: list[SignalEvent] = []

        meal_hits = _match_any(message, MEAL_PATTERNS)
        food_hit = FOOD_ITEMS.search(message)

        # --- Meal detection (last_meal) ---
        if meal_hits or food_hit:
            source = meal_hits[0] if meal_hits else food_hit.group()
            events.append(SignalEvent(
                signal_type=SignalType.NUTRITION,
                dimension="nutrition.last_meal",
                value=1.0,  # 1.0 = meal detected at this timestamp
                confidence=0.8 if meal_hits else 0.6,  # explicit mention > food item inference
                source_text=source,
                timestamp=ts,
            ))

        # --- Meal quality ---
        high_hits = _match_any(message, HIGH_QUALITY_PATTERNS)
        low_hits = _match_any(message, LOW_QUALITY_PATTERNS)

        if high_hits:
            events.append(SignalEvent(
                signal_type=SignalType.NUTRITION,
                dimension="nutrition.quality",
                value=0.85,
                confidence=0.7,
                source_text=high_hits[0],
                timestamp=ts,
            ))
        elif low_hits:
            events.append(SignalEvent(
                signal_type=SignalType.NUTRITION,
                dimension="nutrition.quality",
                value=0.25,
                confidence=0.7,
                source_text=low_hits[0],
                timestamp=ts,
            ))

        # --- Hunger / skipping ---
        hunger_hits = _match_any(message, HUNGER_PATTERNS)
        if hunger_hits:
            events.append(SignalEvent(
                signal_type=SignalType.NUTRITION,
                dimension="nutrition.last_meal",
                value=0.0,  # 0.0 = no recent meal / hunger signal
                confidence=0.75,
                source_text=hunger_hits[0],
                timestamp=ts,
            ))
            events.append(SignalEvent(
                signal_type=SignalType.NUTRITION,
                dimension="nutrition.quality",
                value=0.1,
                confidence=0.6,
                source_text=hunger_hits[0],
                timestamp=ts,
            ))

        return events
