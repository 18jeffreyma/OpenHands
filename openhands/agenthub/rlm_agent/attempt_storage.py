"""Attempt storage system for RLM agent.

This module provides functionality to store, retrieve, and summarize attempts
made by the RLM agent during its execution.
"""

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.event import Event

if TYPE_CHECKING:
    from openhands.llm.llm import LLM


@dataclass
class Attempt:
    """Represents a single attempt made by the agent."""

    attempt_id: str
    """Unique identifier for this attempt."""

    phase: str
    """Phase during which this attempt was made (ATTEMPT, CHARACTERIZE, or REFLECT)."""

    start_event_id: int
    """Event ID where this attempt started."""

    end_event_id: int | None = None
    """Event ID where this attempt ended (None if still in progress)."""

    summary: str = ''
    """Summary of what was done in this attempt."""

    characterization_summary: str = ''
    """A comprehensive semantic summary characterizing this attempt.

    This includes key modifications, performance details, confidence assessment,
    and other information useful for comparing and identifying attempts during reflection.
    """

    patch: str = ''
    """Git patch/diff representing the changes made in this attempt."""

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
            phase: The phase during which this attempt is made (ATTEMPT, CHARACTERIZE, or REFLECT).
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

    def finish_attempt(
        self,
        end_event_id: int,
        summary: str = '',
        patch: str = '',
        characterization_summary: str = '',
    ) -> None:
        """Finish the current attempt.

        Args:
            end_event_id: The event ID where this attempt ends.
            summary: Optional summary of what was done in this attempt.
            patch: Optional git patch/diff representing the changes made in this attempt.
            characterization_summary: Optional semantic summary for this attempt.
        """
        if self.current_attempt is None:
            return

        self.current_attempt.end_event_id = end_event_id
        if summary:
            self.current_attempt.summary = summary
        if patch:
            self.current_attempt.patch = patch
        if characterization_summary:
            self.current_attempt.characterization_summary = characterization_summary
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
            phase: The phase to filter by (ATTEMPT, CHARACTERIZE, or REFLECT).

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
                    'characterization': attempt.characterization_summary
                    or 'Not characterized',
                    'start_event_id': attempt.start_event_id,
                    'end_event_id': attempt.end_event_id,
                    'num_events': len(attempt.events),
                }
            )
        return summaries

    def get_best_attempt(
        self, llm: 'LLM | None' = None, prompt_dir: str | None = None
    ) -> Attempt | None:
        """Get the best attempt based on LLM reflection or heuristic fallback.

        If an LLM is provided, uses LLM reflection to evaluate which attempt is best
        based on summaries, phase, and other metadata. Otherwise, falls back to
        returning the most recent completed attempt.

        Args:
            llm: Optional LLM instance to use for reflection. If None, uses heuristic.
            prompt_dir: Optional directory containing prompt templates. Required if using LLM reflection.

        Returns:
            The best Attempt object, or None if no attempts exist.
        """
        completed_attempts = [
            attempt for attempt in self.attempts if attempt.end_event_id is not None
        ]
        if not completed_attempts:
            return None

        # If no LLM provided, use heuristic fallback
        if llm is None:
            # Return the most recent completed attempt
            return max(completed_attempts, key=lambda a: a.end_event_id or 0)

        # Use LLM reflection to determine the best attempt
        if prompt_dir is None:
            logger.warning(
                'prompt_dir not provided, cannot use LLM reflection. '
                'Falling back to heuristic.'
            )
            return max(completed_attempts, key=lambda a: a.end_event_id or 0)

        try:
            return self._get_best_attempt_with_reflection(
                completed_attempts, llm, prompt_dir
            )
        except Exception as e:
            logger.warning(
                f'Failed to use LLM reflection for best attempt selection: {e}. '
                'Falling back to heuristic.'
            )
            # Fallback to heuristic
            return max(completed_attempts, key=lambda a: a.end_event_id or 0)

    def _get_best_attempt_with_reflection(
        self, attempts: list[Attempt], llm: 'LLM', prompt_dir: str
    ) -> Attempt | None:
        """Use LLM reflection to determine the best attempt.

        Args:
            attempts: List of completed attempts to evaluate.
            llm: LLM instance to use for reflection.
            prompt_dir: Directory containing prompt templates.

        Returns:
            The best Attempt object according to LLM reflection.
        """
        if not attempts:
            return None

        # If only one attempt, return it
        if len(attempts) == 1:
            return attempts[0]

        # Load the reflection prompt template
        env = Environment(loader=FileSystemLoader(prompt_dir))
        try:
            template = env.get_template('reflection_prompt.j2')
        except Exception as e:
            logger.error(f'Failed to load reflection_prompt.j2 template: {e}')
            raise

        # Format attempts data for the template
        attempts_data = []
        for attempt in attempts:
            attempts_data.append(
                {
                    'attempt_id': attempt.attempt_id,
                    'phase': attempt.phase,
                    'summary': attempt.summary or 'No summary available',
                    'num_events': len(attempt.events),
                    'start_event_id': attempt.start_event_id,
                    'end_event_id': attempt.end_event_id or 'ongoing',
                }
            )

        # Render the reflection prompt using the template
        reflection_prompt = template.render(attempts=attempts_data).strip()

        # Call LLM for reflection
        messages = [
            Message(
                role='user',
                content=[TextContent(text=reflection_prompt)],
            )
        ]

        try:
            response = llm.completion(messages=messages)
            # Extract the attempt ID from the response
            response_text = response.choices[0].message.content.strip()

            # Try to extract attempt ID from response
            # Look for patterns like "attempt-1", "attempt-2", etc.
            attempt_id_match = re.search(r'attempt-(\d+)', response_text)
            if attempt_id_match:
                attempt_id = f"attempt-{attempt_id_match.group(1)}"
                # Find the attempt with this ID
                for attempt in attempts:
                    if attempt.attempt_id == attempt_id:
                        logger.info(
                            f'LLM reflection selected {attempt_id} as the best attempt'
                        )
                        return attempt

            # If we couldn't parse the response, log and fall back
            logger.warning(
                f'Could not parse attempt ID from LLM response: {response_text}. '
                'Falling back to heuristic.'
            )
            return max(attempts, key=lambda a: a.end_event_id or 0)

        except Exception as e:
            logger.error(f'Error during LLM reflection: {e}')
            # Fallback to heuristic
            return max(attempts, key=lambda a: a.end_event_id or 0)



