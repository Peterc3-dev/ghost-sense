"""Tests for the state store / accumulator."""

import time

from ghost_sense.accumulator.state_store import StateStore
from ghost_sense.models import FieldState, SignalEvent, SignalType


class TestStateStore:
    def test_record_and_query_events(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        event = SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.slang_ratio",
            value=0.6,
            confidence=0.85,
            source_text="yo bruh ngl",
            timestamp=time.time(),
        )
        store.record_event(event)
        events = store.get_events(dimension="register.slang_ratio")
        assert len(events) == 1
        assert events[0].value == 0.6
        store.close()

    def test_update_state_creates_snapshot(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        event = SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.formality_score",
            value=0.8,
            confidence=0.9,
            source_text="formal text here",
            timestamp=time.time(),
        )
        store.update_state(event)
        state = store.load_state()
        assert "register.formality_score" in state.fields
        assert state.fields["register.formality_score"].value == 0.8
        store.close()

    def test_state_merges_signals(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        now = time.time()

        store.update_state(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.slang_ratio",
            value=0.8,
            confidence=0.9,
            source_text="first msg",
            timestamp=now,
        ))
        store.update_state(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.slang_ratio",
            value=0.2,
            confidence=0.9,
            source_text="second msg",
            timestamp=now + 1,
        ))

        state = store.load_state()
        # Value should be between 0.2 and 0.8 — merged
        val = state.fields["register.slang_ratio"].value
        assert 0.2 < val < 0.8
        store.close()

    def test_empty_state_on_fresh_db(self, tmp_path):
        store = StateStore(str(tmp_path / "empty.db"))
        state = store.load_state()
        assert len(state.fields) == 0
        store.close()
