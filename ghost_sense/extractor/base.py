"""Abstract base class for all signal extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ghost_sense.models import SignalEvent


class BaseExtractor(ABC):
    """All extractors take a message + metadata, return signal events. No side effects."""

    @abstractmethod
    def extract(self, message: str, metadata: dict) -> list[SignalEvent]:
        """Extract signals from a single message.

        Args:
            message: The raw message text.
            metadata: Must include 'timestamp' (float), 'local_time' (str HH:MM),
                      'message_index_in_session' (int),
                      'time_since_last_message_seconds' (float | None).

        Returns:
            List of SignalEvent instances (may be empty).
        """
        ...
