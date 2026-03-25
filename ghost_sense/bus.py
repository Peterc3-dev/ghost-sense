"""Minimal in-process event bus: extractors emit, accumulator subscribes."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from ghost_sense.models import SignalEvent

Handler = Callable[[SignalEvent], None]


class EventBus:
    """Simple pub/sub keyed by signal type name or '*' for all events."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def subscribe(self, channel: str, handler: Handler) -> None:
        """Subscribe to a channel. Use '*' for all events."""
        self._handlers[channel].append(handler)

    def emit(self, event: SignalEvent) -> None:
        """Emit a signal event to subscribers."""
        for handler in self._handlers.get(event.signal_type.value, []):
            handler(event)
        for handler in self._handlers.get("*", []):
            handler(event)
