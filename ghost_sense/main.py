"""Entry point: process_message -> updated StateVector."""

from __future__ import annotations

import time

from ghost_sense.accumulator.state_store import StateStore
from ghost_sense.bus import EventBus
from ghost_sense.extractor.absence import AbsenceExtractor
from ghost_sense.extractor.cadence import CadenceExtractor
from ghost_sense.extractor.nutrition import NutritionExtractor
from ghost_sense.extractor.register import RegisterExtractor
from ghost_sense.extractor.sleep import SleepExtractor
from ghost_sense.extractor.stress import StressExtractor
from ghost_sense.models import StateVector


class GhostSense:
    """Main engine: wire extractors -> bus -> accumulator."""

    def __init__(self, db_path: str = "ghost_sense.db") -> None:
        self.store = StateStore(db_path)
        self.bus = EventBus()

        # Wire accumulator to bus
        self.bus.subscribe("*", self.store.update_state)

        # Absence extractor is special — it needs signal observations from other extractors
        self.absence_extractor = AbsenceExtractor()

        # All extractors (absence runs last, after others have emitted)
        self.extractors = [
            RegisterExtractor(),
            SleepExtractor(),
            NutritionExtractor(),
            StressExtractor(),
            CadenceExtractor(),
        ]

    def process_message(self, text: str, metadata: dict | None = None) -> StateVector:
        """Process a single message through all extractors and return updated state."""
        if metadata is None:
            metadata = {}
        metadata.setdefault("timestamp", time.time())
        metadata.setdefault("local_time", time.strftime("%H:%M"))
        metadata.setdefault("message_index_in_session", 0)
        metadata.setdefault("time_since_last_message_seconds", None)

        ts = metadata["timestamp"]

        # Run primary extractors
        for extractor in self.extractors:
            events = extractor.extract(text, metadata)
            for event in events:
                self.bus.emit(event)
                # Feed observations to absence tracker
                self.absence_extractor.record_signal(event.dimension, ts)

        # Run absence extractor last (checks for gaps)
        absence_events = self.absence_extractor.extract(text, metadata)
        for event in absence_events:
            self.bus.emit(event)

        return self.store.load_state()

    def get_state(self) -> StateVector:
        """Get current state with decay applied."""
        return self.store.load_state()

    def close(self) -> None:
        self.store.close()
