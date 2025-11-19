"""Attempt storage system for RLM agent.

This module provides functionality to store, retrieve, and summarize attempts
made by the RLM agent during its execution.
"""

from dataclasses import dataclass, field
from typing import Any

from openhands.events.event import Event


@dataclass
class Attempt:
    """Represents a single attempt made by the agent."""

    attempt_id: str
    """Unique identifier for this attempt."""

    phase: str
    """Phase during which this attempt was made (ATTEMPT or RLM)."""

    start_event_id: int
    """Event ID where this attempt started."""

    end_event_id: int | None = None
    """Event ID where this attempt ended (None if still in progress)."""

    summary: str = ''
    """Summary of what was done in this attempt."""

    events: list[Event] = field(default_factory=list)
    """Events that occurred during this attempt."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about this attempt."""


class AttemptStorage:
    """Stores and manages attempts made by the RLM agent."""

    def __init__(self):
        self.attempts: list[Attempt] = []
        self.current_attempt: Attempt | None = None
        self._next_attempt_id = 1

    def start_attempt(self, phase: str, start_event_id: int) -> Attempt:
        """Start a new attempt.

        Args:
            phase: The phase during which this attempt is made (ATTEMPT or RLM).
            start_event_id: The event ID where this attempt starts.

        Returns:
            The newly created Attempt object.
        """
        attempt_id = f'attempt-{self._next_attempt_id}'
        self._next_attempt_id += 1

        attempt = Attempt(
            attempt_id=attempt_id,
            phase=phase,
            start_event_id=start_event_id,
        )
        self.attempts.append(attempt)
        self.current_attempt = attempt
        return attempt

    def finish_attempt(self, end_event_id: int, summary: str = '') -> None:
        """Finish the current attempt.

        Args:
            end_event_id: The event ID where this attempt ends.
            summary: Optional summary of what was done in this attempt.
        """
        if self.current_attempt is None:
            return

        self.current_attempt.end_event_id = end_event_id
        if summary:
            self.current_attempt.summary = summary
        self.current_attempt = None

    def add_event_to_current_attempt(self, event: Event) -> None:
        """Add an event to the current attempt.

        Args:
            event: The event to add.
        """
        if self.current_attempt is not None:
            self.current_attempt.events.append(event)

    def get_attempt(self, attempt_id: str) -> Attempt | None:
        """Get an attempt by its ID.

        Args:
            attempt_id: The ID of the attempt to retrieve.

        Returns:
            The Attempt object if found, None otherwise.
        """
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        return None

    def get_attempts_by_phase(self, phase: str) -> list[Attempt]:
        """Get all attempts made during a specific phase.

        Args:
            phase: The phase to filter by (ATTEMPT or RLM).

        Returns:
            List of Attempt objects from the specified phase.
        """
        return [attempt for attempt in self.attempts if attempt.phase == phase]

    def get_all_attempts(self) -> list[Attempt]:
        """Get all attempts.

        Returns:
            List of all Attempt objects.
        """
        return self.attempts.copy()

    def get_summarized_attempts(self) -> list[dict[str, Any]]:
        """Get a summarized view of all attempts.

        Returns:
            List of dictionaries containing summarized attempt information.
        """
        summaries = []
        for attempt in self.attempts:
            summaries.append(
                {
                    'id': attempt.attempt_id,
                    'phase': attempt.phase,
                    'summary': attempt.summary or 'No summary available',
                    'start_event_id': attempt.start_event_id,
                    'end_event_id': attempt.end_event_id,
                    'num_events': len(attempt.events),
                }
            )
        return summaries

    def get_best_attempt(self) -> Attempt | None:
        """Get the best attempt based on some heuristic.

        Currently returns the most recent completed attempt.

        Returns:
            The best Attempt object, or None if no attempts exist.
        """
        completed_attempts = [
            attempt for attempt in self.attempts if attempt.end_event_id is not None
        ]
        if not completed_attempts:
            return None

        # Return the most recent completed attempt
        return max(completed_attempts, key=lambda a: a.end_event_id or 0)



