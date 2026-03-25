"""Tests for nutrition extractor."""

import time

import pytest

from ghost_sense.extractor.nutrition import NutritionExtractor
from ghost_sense.models import SignalType


@pytest.fixture
def extractor():
    return NutritionExtractor()


def _meta(**kw):
    m = {
        "timestamp": time.time(),
        "local_time": "12:00",
        "message_index_in_session": 0,
        "time_since_last_message_seconds": None,
    }
    m.update(kw)
    return m


class TestNutritionExtractor:
    def test_explicit_meal(self, extractor):
        events = extractor.extract("just had lunch", _meta())
        meal = [e for e in events if e.dimension == "nutrition.last_meal" and e.value > 0.5]
        assert len(meal) == 1

    def test_food_item_detection(self, extractor):
        events = extractor.extract("thinking about getting some ramen", _meta())
        meal = [e for e in events if e.dimension == "nutrition.last_meal"]
        assert len(meal) >= 1

    def test_cooking(self, extractor):
        events = extractor.extract("just cooked a big batch of pasta", _meta())
        meal = [e for e in events if e.dimension == "nutrition.last_meal"]
        assert len(meal) >= 1

    def test_high_quality_meal(self, extractor):
        events = extractor.extract("had a clean home-cooked meal with grilled chicken and veggies", _meta())
        quality = [e for e in events if e.dimension == "nutrition.quality"]
        assert len(quality) >= 1
        assert quality[0].value > 0.7

    def test_low_quality_meal(self, extractor):
        events = extractor.extract("grabbed mcdonalds on the way", _meta())
        quality = [e for e in events if e.dimension == "nutrition.quality"]
        assert len(quality) >= 1
        assert quality[0].value < 0.4

    def test_hunger_signal(self, extractor):
        events = extractor.extract("starving, haven't eaten all day", _meta())
        meal = [e for e in events if e.dimension == "nutrition.last_meal" and e.value < 0.5]
        assert len(meal) >= 1

    def test_skipped_meal(self, extractor):
        events = extractor.extract("skipped breakfast again", _meta())
        meal = [e for e in events if e.dimension == "nutrition.last_meal" and e.value < 0.5]
        assert len(meal) >= 1

    def test_no_nutrition_in_unrelated(self, extractor):
        events = extractor.extract("the function signature needs refactoring", _meta())
        assert len(events) == 0

    def test_all_events_are_nutrition_type(self, extractor):
        events = extractor.extract("just ate a huge breakfast, eggs and toast", _meta())
        assert all(e.signal_type == SignalType.NUTRITION for e in events)

    def test_fast_food_quality(self, extractor):
        events = extractor.extract("grabbed some fast food because no time", _meta())
        quality = [e for e in events if e.dimension == "nutrition.quality"]
        assert len(quality) >= 1
        assert quality[0].value < 0.4

    def test_ordered_food(self, extractor):
        events = extractor.extract("ordered pizza for everyone", _meta())
        meal = [e for e in events if e.dimension == "nutrition.last_meal" and e.value > 0.5]
        assert len(meal) >= 1
