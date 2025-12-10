"""RLM Agent - Recursive Language Model Agent.

This agent alternates between three phases (ATTEMPT, CHARACTERIZE, REFLECT)
over multiple iterations to solve tasks iteratively.
"""

import json
import os
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam
    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.rlm_agent.function_calling as rlm_function_calling
from openhands.agenthub.codeact_agent.tools.bash import create_cmd_run_tool
from openhands.agenthub.codeact_agent.tools.browser import BrowserTool
from openhands.agenthub.codeact_agent.tools.condensation_request import (
    CondensationRequestTool,
)
from openhands.agenthub.codeact_agent.tools.ipython import IPythonTool
from openhands.agenthub.codeact_agent.tools.llm_based_edit import LLMBasedFileEditTool
from openhands.agenthub.codeact_agent.tools.str_replace_editor import (
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.task_tracker import (
    create_task_tracker_tool,
)
from openhands.agenthub.codeact_agent.tools.think import ThinkTool
from openhands.agenthub.rlm_agent.attempt_storage import AttemptStorage
from openhands.agenthub.rlm_agent.tools.attempt import FinishAttemptTool
from openhands.agenthub.rlm_agent.tools.browse_attempt import (
    BrowseAttemptTool,
)
from openhands.agenthub.rlm_agent.tools.finish_characterization import (
    FinishCharacterizationTool,
)
from openhands.agenthub.rlm_agent.tools.finish_reflection import (
    FinishReflectionTool,
)
from openhands.agenthub.rlm_agent.tools.submit_attempt_as_final import (
    SubmitAttemptAsFinalTool,
)
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import (
    Action,
    AgentFinishAction,
    CmdRunAction,
    ExpandPreviousAttemptAction,  # Keep class name for compatibility, but tool is now browse_attempt
    FinishAttemptAction,
    FinishCharacterizationAction,
    FinishReflectionAction,
    MessageAction,
    SubmitAttemptAsFinalAction,
)
from openhands.events.action.agent import CondensationAction
from openhands.events.event import Event, EventSource
from openhands.events.observation import (
    CmdOutputObservation,
    ExpandPreviousAttemptObservation,  # Keep class name for compatibility, but tool is now browse_attempt
    FinishAttemptObservation,
    SubmitAttemptAsFinalObservation,
)
from openhands.llm.llm_utils import check_tools
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from openhands.utils.prompt import PromptManager


class Phase(str, Enum):
    """Represents the current phase of the RLM agent."""

    ATTEMPT = 'ATTEMPT'
    CHARACTERIZE = 'CHARACTERIZE'
    REFLECT = 'REFLECT'




class RLMAgent(Agent):
    """RLM Agent that alternates between ATTEMPT, CHARACTERIZE, and REFLECT phases.

    Flow: ATTEMPT -> CHARACTERIZE -> REFLECT -> ATTEMPT (new iteration)
    After max_iterations, selects and applies the best attempt.
    """

    VERSION = '2.0'

    sandbox_plugins: list[PluginRequirement] = [
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initialize RLM Agent.

        Args:
            config: Agent configuration
            llm_registry: LLM registry for getting LLM instances
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque[Action] = deque()

        # RLM-specific state
        self.attempt_storage = AttemptStorage()
        self.current_phase = Phase.ATTEMPT
        self.characterize_phase_start_event_id: int | None = None
        self.reflect_phase_start_event_id: int | None = None

        # Get max_iterations from config or environment
        rlm_max_iterations = None
        rlm_phase_log_dir = None
        if hasattr(config, 'extended') and config.extended:
            try:
                rlm_max_iterations = config.extended.get('rlm_max_iterations')
                rlm_phase_log_dir = config.extended.get('rlm_phase_log_dir')
            except (KeyError, AttributeError):
                pass

        # Fall back to global agent config max_iterations from the runtime/config,
        # then to environment. This ensures values provided by run_infer_rlm.sh
        # (passed via OpenHandsConfig.max_iterations) are honored even if the
        # extended section is missing.
        if rlm_max_iterations is None and getattr(config, 'max_iterations', None) is not None:
            rlm_max_iterations = config.max_iterations
        if rlm_max_iterations is None:
            rlm_max_iterations = int(os.environ.get('RLM_MAX_ITERATIONS', '3'))
        self.max_iterations: int = int(rlm_max_iterations)
        self.current_iteration: int = 0
        self._no_tool_retry_counts: dict[str, int] = {}
        self._no_tool_prompted: set[str] = set()

        # Phase logging directory (set via extended config)
        self.phase_log_dir: str | None = rlm_phase_log_dir
        self._phase_step_counter: dict[str, int] = {}  # Track step count per phase

        # Patch extraction/application state
        self._pending_patch_extraction: tuple[int, int] | None = None
        self._pending_patch_application: bool = False
        self._apply_patch_action_id: int | None = None

        # Initialize tools and memory
        self.tools = self._get_tools()
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)
        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        self.llm = self.llm_registry.get_router(self.config)

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager for current phase."""
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
                system_prompt_filename=self._get_system_prompt_filename(),
            )
        return self._prompt_manager

    def _get_system_prompt_filename(self) -> str:
        """Get system prompt filename based on current phase."""
        if self.current_phase == Phase.ATTEMPT:
            return 'system_prompt_attempt.j2'
        elif self.current_phase == Phase.CHARACTERIZE:
            return 'system_prompt_characterize.j2'
        else:  # Phase.REFLECT
            return 'system_prompt_reflect.j2'

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        """Get available tools for current phase and browsing state."""
        tools = []

        if self.current_phase == Phase.ATTEMPT:
            # Normal attempt mode: full toolset
            if self.config.enable_cmd:
                tools.append(create_cmd_run_tool(use_short_description=False))
            if self.config.enable_think:
                tools.append(ThinkTool)
            if self.config.enable_browsing:
                import sys
                if sys.platform != 'win32':
                    tools.append(BrowserTool)
            if self.config.enable_jupyter:
                tools.append(IPythonTool)
            if self.config.enable_plan_mode:
                tools.append(create_task_tracker_tool(use_short_description=False))
            if self.config.enable_llm_editor:
                tools.append(LLMBasedFileEditTool)
            elif self.config.enable_editor:
                tools.append(
                    create_str_replace_editor_tool(
                        use_short_description=False,
                        runtime_type=self.config.runtime,
                    )
                )
            if self.config.enable_condensation_request:
                tools.append(CondensationRequestTool)
            tools.extend([
                FinishAttemptTool,
            ])

        elif self.current_phase == Phase.CHARACTERIZE:
            if self.config.enable_cmd:
                tools.append(create_cmd_run_tool(use_short_description=False))
            if self.config.enable_think:
                tools.append(ThinkTool)
            tools.append(FinishCharacterizationTool)

        else:  # Phase.REFLECT
            tools.extend([
                BrowseAttemptTool,  # Renamed from expand_previous_attempt
                FinishReflectionTool,
                SubmitAttemptAsFinalTool,
            ])
            if self.config.enable_think:
                tools.append(ThinkTool)

        return tools

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self.pending_actions.clear()
        self.attempt_storage = AttemptStorage()
        self.current_phase = Phase.ATTEMPT
        self.characterize_phase_start_event_id = None
        self.reflect_phase_start_event_id = None
        self.current_iteration = 0
        self._pending_patch_extraction = None
        self._pending_patch_application = False
        self._apply_patch_action_id = None
        self._phase_step_counter = {}  # Reset step counter
        self._no_tool_retry_counts = {}
        self._no_tool_prompted = set()

    def _transition_to_next_phase(self, state: State) -> None:
        """Transition to the next phase: ATTEMPT -> CHARACTERIZE -> REFLECT -> ATTEMPT.

        Per spec: Each phase transition should:
        1. Update the phase state
        2. Update the system prompt (via prompt_manager)
        3. Update available tools
        4. Reset context appropriately (REFLECT phase gets fresh context)
        """
        if self.current_phase == Phase.ATTEMPT:
            self.current_phase = Phase.CHARACTERIZE
            if state.history:
                self.characterize_phase_start_event_id = state.history[-1].id + 1
            logger.info('Transitioning to CHARACTERIZE phase')

        elif self.current_phase == Phase.CHARACTERIZE:
            self.current_phase = Phase.REFLECT
            self.characterize_phase_start_event_id = None
            if state.history:
                self.reflect_phase_start_event_id = state.history[-1].id + 1
            logger.info('Transitioning to REFLECT phase')

        else:  # Phase.REFLECT
            self.current_phase = Phase.ATTEMPT
            self.reflect_phase_start_event_id = None
            self.current_iteration += 1
            logger.info(f'Transitioning to ATTEMPT phase (iteration {self.current_iteration})')

        # Update tools and prompt manager for new phase
        # Per spec: Each phase has its own system prompt
        self.tools = self._get_tools()
        self._prompt_manager = None  # Force prompt manager to reload with new phase's system prompt
        # Ensure conversation_memory uses the updated prompt manager
        # This is critical for REFLECT phase to use system_prompt_reflect.j2
        self.conversation_memory.prompt_manager = self.prompt_manager

    def _should_finish(self) -> bool:
        """Check if we should finish and return best solution."""
        return self.current_iteration >= self.max_iterations

    def _create_message_action(
        self, content: str, wait_for_response: bool = False, source: EventSource = EventSource.USER
    ) -> MessageAction:
        """Create a MessageAction with specified content and source."""
        if content is None:
            content = ''
        elif not isinstance(content, str):
            content = str(content)
        action = MessageAction(content=content, wait_for_response=wait_for_response)
        action._source = source  # type: ignore
        return action

    def _get_characterize_transition_message(self, attempt_summary: str) -> str:
        """Generate transition message for CHARACTERIZE phase."""
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        env = Environment(loader=FileSystemLoader(prompt_dir))
        template = env.get_template('characterize_transition.j2')
        base = template.render(attempt_summary=attempt_summary or 'No summary provided.')

        # Optional task overview provided by evaluation harness (e.g., swefficiency).
        overview = None
        if hasattr(self.config, 'extended') and self.config.extended:
            overview = self.config.extended.get('conversation_instruction')

        if overview:
            return f'{overview}\n\n{base}'
        return base

    def _get_reflect_transition_messages(self) -> tuple[str, str]:
        """Generate transition messages for REFLECT phase.

        Returns:
            Tuple of (summary_message, task_message) - two separate messages
        """
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        env = Environment(loader=FileSystemLoader(prompt_dir))
        summaries = self.attempt_storage.get_summarized_attempts(completed_only=True)
        summary_template = env.get_template('reflect_transition.j2')
        task_template = env.get_template('reflect_transition_task.j2')
        summary_message = summary_template.render(attempt_summaries=summaries)
        task_message = task_template.render()
        return (summary_message, task_message)

    def _build_characterize_context(self, state: State) -> list[Message]:
        """Build context for CHARACTERIZE phase with attempt events + CHARACTERIZE phase events.

        Note: This method uses state.history (full history) instead of condensed_history because:
        1. We need ALL events from the completed attempt to properly analyze what was done
        2. Condensation may have removed some attempt events that are needed for characterization
        3. We perform our own precise filtering based on event IDs (attempt range + CHARACTERIZE phase)

        The filtered events are then passed to conversation_memory.process_events as condensed_history,
        which will handle message formatting and any further processing.
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Find the most recently completed ATTEMPT phase attempt
        completed_attempts = [
            attempt for attempt in self.attempt_storage.attempts
            if attempt.end_event_id is not None and attempt.phase == Phase.ATTEMPT
        ]

        # Edge case: CHARACTERIZE phase should always have a completed attempt
        # This should never happen in normal flow, but log a warning if it does
        if not completed_attempts:
            logger.warning(
                'CHARACTERIZE phase entered but no completed ATTEMPT phase attempts found. '
                'This may indicate a state inconsistency. Proceeding with empty event list.'
            )

        characterize_events: list[Event] = []
        transition_action: MessageAction | None = None
        attempt_summary = ''

        if completed_attempts:
            # Get the most recent completed ATTEMPT phase attempt
            most_recent_attempt = max(completed_attempts, key=lambda a: a.end_event_id or 0)
            attempt_start_id = most_recent_attempt.start_event_id
            attempt_end_id = most_recent_attempt.end_event_id
            attempt_summary = most_recent_attempt.summary or ''

            # Include events from the attempt (to analyze what was done)
            # plus CHARACTERIZE phase events ONLY - exclude any events in between
            # (e.g., from REFLECT phase of previous iterations)
            characterize_start_id = self.characterize_phase_start_event_id
            if characterize_start_id is not None and attempt_end_id is not None:
                for event in state.history:
                    # Include events from the attempt (inclusive) OR from CHARACTERIZE phase start onwards
                    # STRICT: Only include if within attempt range OR at/after CHARACTERIZE start
                    # This excludes events between attempt_end_id and characterize_start_id
                    in_attempt_range = attempt_start_id <= event.id <= attempt_end_id
                    in_characterize_phase = event.id >= characterize_start_id
                    if in_attempt_range or in_characterize_phase:
                        characterize_events.append(event)
                        # Only pick up transition_action if it's a USER-sourced MessageAction
                        # in the CHARACTERIZE phase that contains the attempt summary
                        if transition_action is None and isinstance(event, MessageAction) and \
                           event.id >= characterize_start_id and event.source == EventSource.USER:
                            # Check if this transition message already includes the attempt summary
                            # If it does, use it; otherwise, we'll create a new one with the summary
                            if attempt_summary and attempt_summary in event.content:
                                transition_action = event
                            elif not attempt_summary:
                                # No summary available, use existing transition if found
                                transition_action = event
            else:
                # Fallback: include all events from the attempt onwards
                for event in state.history:
                    if event.id >= attempt_start_id:
                        characterize_events.append(event)
                        if transition_action is None and isinstance(event, MessageAction) and \
                           event.source == EventSource.USER:
                            if attempt_summary and attempt_summary in event.content:
                                transition_action = event
                            elif not attempt_summary:
                                transition_action = event

        # Always create/update transition message to ensure attempt summary is included
        # Per spec: CHARACTERIZE phase should receive the attempt summary in the transition message
        if transition_action is None or (attempt_summary and attempt_summary not in transition_action.content):
            # Create transition message with attempt summary
            transition_content = self._get_characterize_transition_message(attempt_summary)
            transition_action = MessageAction(content=transition_content)
            transition_action._source = EventSource.USER  # type: ignore

        # Ensure prompt manager is using the CHARACTERIZE phase system prompt
        # This should already be set in _transition_to_next_phase, but ensure it's correct
        if self.current_phase != Phase.CHARACTERIZE:
            logger.warning('_build_characterize_context called but current_phase is not CHARACTERIZE')
        self.conversation_memory.prompt_manager = self.prompt_manager

        messages = self.conversation_memory.process_events(
            condensed_history=characterize_events,
            initial_user_action=transition_action,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def _build_reflect_context(self, state: State) -> list[Message]:
        """Build fresh context for REFLECT phase with only attempt summaries.

        Per spec: REFLECT phase should have:
        - Fresh context with only REFLECT phase events (summaries of all attempts)
        - System prompt: system_prompt_reflect.j2
        - No previous trajectory context (reset)
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Per spec: Fresh context with only REFLECT phase events
        # Filter events to ONLY include REFLECT phase events (reset trajectory context)
        reflect_events: list[Event] = []
        summary_action: MessageAction | None = None
        task_action: MessageAction | None = None

        if self.reflect_phase_start_event_id is not None:
            # Only include events from REFLECT phase start onwards
            # This ensures we have a fresh context without previous attempt/characterize events
            for event in state.history:
                if event.id >= self.reflect_phase_start_event_id:
                    reflect_events.append(event)
                    # Look for the two transition messages (summary and task)
                    if isinstance(event, MessageAction) and event.source == EventSource.USER:
                        summary_content, task_content = self._get_reflect_transition_messages()
                        # Identify which message this is based on content
                        if summary_content in event.content or (not summary_action and len(event.content) < 500):
                            summary_action = event
                        elif task_content in event.content or len(event.content) > 200:
                            task_action = event

        # Create transition messages if not found
        # Per spec: Transition messages should be split - first summary, then task
        summary_content, task_content = self._get_reflect_transition_messages()
        if summary_action is None:
            # First message: attempt summaries only
            summary_action = MessageAction(content=summary_content)
            summary_action._source = EventSource.USER  # type: ignore

        # Ensure prompt manager is using the REFLECT phase system prompt
        # This should already be set in _transition_to_next_phase, but ensure it's correct
        if self.current_phase != Phase.REFLECT:
            logger.warning('_build_reflect_context called but current_phase is not REFLECT')
        self.conversation_memory.prompt_manager = self.prompt_manager

        # Process events with fresh context (only REFLECT phase events)
        # This ensures trajectory context is reset per spec
        # Use summary message as initial user action
        messages = self.conversation_memory.process_events(
            condensed_history=reflect_events,
            initial_user_action=summary_action,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        # Add second message (task instructions) after the summary message
        # If task_action was found in events, it should already be in messages
        # Otherwise, add it manually
        if task_action is None:
            # Add task message as a separate user message
            messages.append(Message(role='user', content=[TextContent(text=task_content)]))

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def _is_gemini_model(self) -> bool:
        """Check if current LLM is a Gemini model."""
        return 'gemini' in self.llm.config.model.lower()

    def _actions_have_tools(self, actions: list[Action]) -> bool:
        """Determine if any action is a tool call (non-MessageAction)."""
        return any(not isinstance(action, MessageAction) for action in actions)

    def _phase_retry_key(self) -> str:
        """Generate a retry key scoped to phase + iteration."""
        return f'{self.current_phase.value}:{self.current_iteration}'

    def _phase_requires_tool_completion(self) -> bool:
        """Phases that should respond with finish_* or submit_* tools only."""
        return self.current_phase in {Phase.CHARACTERIZE, Phase.REFLECT}

    def _get_tool_call_nudge(self) -> str:
        """Short corrective prompt to force required tool usage."""
        if self.current_phase == Phase.CHARACTERIZE:
            return (
                'Use the finish_characterization tool with a characterization_summary. '
                'Plain text is not accepted in this phase.'
            )
        if self.current_phase == Phase.REFLECT:
            return (
                'Respond with finish_reflection (or submit_attempt_as_final) containing your plan. '
                'Do not return plain text.'
            )
        return (
            'Use the available tools to make progress and call finish_attempt with a summary when done. '
            'Plain text with no tool calls cannot advance the attempt.'
        )

    def _should_force_tool_call(self, actions: list[Action]) -> bool:
        """Decide whether to reprompt the LLM to issue the required tool call."""
        key = self._phase_retry_key()
        retries = self._no_tool_retry_counts.get(key, 0)

        if self._actions_have_tools(actions):
            self._no_tool_retry_counts.pop(key, None)
            self._no_tool_prompted.discard(key)
            return False

        # CHARACTERIZE/REFLECT accept only tool calls; always nudge when missing.
        if self._phase_requires_tool_completion():
            if retries >= 2:
                logger.warning(
                    f'No tool call in {self.current_phase.value} after {retries} retries; '
                    'continuing without additional nudges.'
                )
                return False
            self._no_tool_retry_counts[key] = retries + 1
            return True

        # ATTEMPT: only nudge on empty/whitespace replies with no tools.
        text_content = ''
        for action in actions:
            if isinstance(action, MessageAction) and action.content:
                text_content += action.content
        if text_content.strip():
            self._no_tool_retry_counts.pop(key, None)
            self._no_tool_prompted.discard(key)
            return False
        if retries >= 2:
            logger.warning(
                f'Empty assistant reply without tools in ATTEMPT after {retries} retries.'
            )
            return False
        self._no_tool_retry_counts[key] = retries + 1
        return True

    def _has_completed_attempt(self) -> bool:
        """Return True if at least one ATTEMPT has end_event_id set."""
        return any(
            attempt.end_event_id is not None and attempt.phase == Phase.ATTEMPT
            for attempt in self.attempt_storage.attempts
        )

    def _has_characterized_attempt(self) -> bool:
        """Return True if a completed ATTEMPT has a characterization_summary."""
        return any(
            attempt.end_event_id is not None
            and attempt.phase == Phase.ATTEMPT
            and bool(attempt.characterization_summary)
            for attempt in self.attempt_storage.attempts
        )

    def _reset_phase(self, phase: Phase) -> None:
        """Force the agent to a specific phase and refresh prompt/tools."""
        self.current_phase = phase
        self.tools = self._get_tools()
        self._prompt_manager = None
        self.conversation_memory.prompt_manager = self.prompt_manager

    def _handle_finish_attempt(self, state: State, event: FinishAttemptAction) -> Action | None:
        """Handle finish_attempt action - extract patch and transition to CHARACTERIZE."""
        # Normalize summary from finish_attempt tool call
        summary = event.message or ''

        # Check if we're waiting for patch extraction
        if self._pending_patch_extraction is not None:
            if self._pending_patch_extraction[0] == event.id:
                # Check for patch extraction result
                for obs_event in reversed(state.history[-5:]):
                    if (
                        isinstance(obs_event, CmdOutputObservation)
                        and obs_event.cause == self._pending_patch_extraction[1]
                    ):
                        patch = obs_event.content
                        if self.attempt_storage.current_attempt:
                            self.attempt_storage.finish_attempt(
                                end_event_id=event.id,
                                summary=summary,
                                patch=patch,
                            )
                        self._pending_patch_extraction = None
                        self._transition_to_next_phase(state)
                        transition_message = self._get_characterize_transition_message(summary)
                        return self._create_message_action(content=transition_message, wait_for_response=False)
                # Still waiting for patch extraction
                return self._create_message_action(content='Extracting patch for attempt...', wait_for_response=False)

        # Extract patch if configured
        extract_cmd = None
        if hasattr(self.config, 'extended') and self.config.extended:
            extended_dict = self.config.extended.model_dump()
            extract_cmd = extended_dict.get('rlm_extract_patch_cmd')

        if extract_cmd and self.attempt_storage.current_attempt:
            extract_action = CmdRunAction(command=extract_cmd)
            extract_action.set_hard_timeout(600)
            self.pending_actions.append(extract_action)
            self._pending_patch_extraction = (event.id, extract_action.id)
            logger.info(f'Extracting patch for attempt {self.attempt_storage.current_attempt.attempt_id}')
            return extract_action

        # No patch extraction, finish attempt and transition
        if self.attempt_storage.current_attempt:
            self.attempt_storage.finish_attempt(end_event_id=event.id, summary=summary)
        self._transition_to_next_phase(state)
        transition_message = self._get_characterize_transition_message(summary)
        return self._create_message_action(content=transition_message, wait_for_response=False)

    def _handle_finish_characterization(self, state: State, event: FinishCharacterizationAction) -> Action:
        """Handle finish_characterization action - transition to REFLECT."""
        # Find the most recently completed attempt (the one that was just finished in ATTEMPT phase)
        # current_attempt is None at this point because finish_attempt was already called
        completed_attempts = [
            attempt for attempt in self.attempt_storage.attempts
            if attempt.end_event_id is not None and attempt.phase == Phase.ATTEMPT
        ]
        if completed_attempts:
            # Get the most recent completed ATTEMPT phase attempt
            most_recent_attempt = max(completed_attempts, key=lambda a: a.end_event_id or 0)
            most_recent_attempt.characterization_summary = event.characterization_summary
        self._transition_to_next_phase(state)
        # Get both transition messages (summary and task)
        summary_message, task_message = self._get_reflect_transition_messages()
        # Queue the task message as a pending action so it follows the summary message
        self.pending_actions.append(
            self._create_message_action(content=task_message, wait_for_response=False)
        )
        # Return the summary message as the first transition message
        return self._create_message_action(content=summary_message, wait_for_response=False)

    def _handle_finish_reflection(self, state: State, event: FinishReflectionAction) -> Action:
        """Handle finish_reflection action - transition to next ATTEMPT."""
        # Normal REFLECT phase finish - transition to new ATTEMPT
        self._transition_to_next_phase(state)
        if state.history:
            start_event_id = state.history[-1].id + 1
        else:
            start_event_id = 0
        self.attempt_storage.start_attempt(phase=Phase.ATTEMPT, start_event_id=start_event_id)

        reflection_insights = event.message if event.message else ''
        if reflection_insights:
            msg = self._create_message_action(
                content=(
                    'Reflection phase completed. From reflection on previous attempts, you should try the following for this attempt:\n\n'
                    f'{reflection_insights}\n\n'
                    f'Starting new attempt phase (iteration {self.current_iteration}).'
                ),
                wait_for_response=False,
            )
            msg._source = EventSource.USER  # ensure ATTEMPT picks this up as the latest user message
            return msg

        msg = self._create_message_action(
            content=f'Reflection phase completed. Starting new attempt phase (iteration {self.current_iteration}).',
            wait_for_response=False,
        )
        msg._source = EventSource.USER  # ensure ATTEMPT picks this up as the latest user message
        return msg

    def _handle_submit_attempt_as_final(self, state: State, event: SubmitAttemptAsFinalAction) -> Action:
        """Handle submit_attempt_as_final action - apply patch and finish."""
        attempt = self.attempt_storage.get_attempt(event.attempt_id)
        if not attempt:
            logger.warning(f'Attempt {event.attempt_id} not found, cannot submit as final')
            return self._create_message_action(
                content=f'Attempt {event.attempt_id} not found. Cannot submit as final.',
                wait_for_response=False,
            )

        logger.info(f'Submitting attempt {event.attempt_id} as final solution')

        # Check if patch application is pending
        if self._pending_patch_application:
            for obs_event in reversed(state.history[-5:]):
                if (
                    isinstance(obs_event, CmdOutputObservation)
                    and self._apply_patch_action_id is not None
                    and obs_event.cause == self._apply_patch_action_id
                ):
                    self._pending_patch_application = False
                    return AgentFinishAction(
                        final_thought=f'Submitted attempt {event.attempt_id} as final solution. {event.message}. Applied patch successfully.'
                    )
            return self._create_message_action(
                content='Applying patch for submitted attempt...',
                wait_for_response=False,
            )

        # Apply patch if configured
        apply_cmd = None
        if hasattr(self.config, 'extended') and self.config.extended:
            extended_dict = self.config.extended.model_dump()
            apply_cmd = extended_dict.get('rlm_apply_patch_cmd')

        if attempt.patch and apply_cmd:
            import base64
            patch_b64 = base64.b64encode(attempt.patch.encode()).decode()
            apply_cmd_final = apply_cmd.replace('{patch}', patch_b64)
            apply_action = CmdRunAction(command=apply_cmd_final)
            apply_action.set_hard_timeout(600)
            self.pending_actions.append(apply_action)
            self._pending_patch_application = True
            self._apply_patch_action_id = apply_action.id
            logger.info(f'Applying patch from submitted attempt {event.attempt_id}')
            return apply_action
        else:
            return AgentFinishAction(
                final_thought=f'Submitted attempt {event.attempt_id} as final solution. {event.message}'
            )

    def _handle_best_attempt_finish(self, state: State) -> Action:
        """Handle finishing after max iterations - select and apply best attempt."""
        # Check if patch application is pending
        if self._pending_patch_application:
            for event in reversed(state.history[-5:]):
                if (
                    isinstance(event, CmdOutputObservation)
                    and self._apply_patch_action_id is not None
                    and event.cause == self._apply_patch_action_id
                ):
                    self._pending_patch_application = False
                    prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
                    best_attempt = self.attempt_storage.get_best_attempt(
                        llm=self.llm, prompt_dir=prompt_dir
                    )
                    if best_attempt:
                        return AgentFinishAction(
                            final_thought=f'Completed {self.current_iteration} iterations. Best attempt: {best_attempt.attempt_id}. {best_attempt.summary}. Applied patch successfully.'
                        )
                    else:
                        return AgentFinishAction(
                            final_thought=f'Completed {self.current_iteration} iterations. No completed attempts found.'
                        )
            return self._create_message_action(
                content='Applying best attempt patch...',
                wait_for_response=False,
            )

        # Get best attempt and apply patch if configured
        prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        best_attempt = self.attempt_storage.get_best_attempt(llm=self.llm, prompt_dir=prompt_dir)

        apply_cmd = None
        if hasattr(self.config, 'extended') and self.config.extended:
            extended_dict = self.config.extended.model_dump()
            apply_cmd = extended_dict.get('rlm_apply_patch_cmd')

        if best_attempt and best_attempt.patch and apply_cmd:
            import base64
            patch_b64 = base64.b64encode(best_attempt.patch.encode()).decode()
            apply_cmd_final = apply_cmd.replace('{patch}', patch_b64)
            apply_action = CmdRunAction(command=apply_cmd_final)
            apply_action.set_hard_timeout(600)
            self.pending_actions.append(apply_action)
            self._pending_patch_application = True
            self._apply_patch_action_id = apply_action.id
            logger.info(f'Applying patch from best attempt {best_attempt.attempt_id}')
            return apply_action
        elif best_attempt:
            return AgentFinishAction(
                final_thought=f'Completed {self.current_iteration} iterations. Best attempt: {best_attempt.attempt_id}. {best_attempt.summary}'
            )
        else:
            return AgentFinishAction(
                final_thought=f'Completed {self.current_iteration} iterations. No completed attempts found.'
            )

    def _enrich_attempt_observations(self, condensed_history: list[Event], state: State) -> None:
        """Enrich ExpandPreviousAttemptObservation (now browse_attempt) with full attempt trajectory."""
        for event in condensed_history:
            if isinstance(event, ExpandPreviousAttemptObservation):
                attempt_id = None
                for prev_event in reversed(state.history):
                    if isinstance(prev_event, ExpandPreviousAttemptAction):
                        if prev_event.id == event.cause:
                            attempt_id = prev_event.attempt_id
                            break

                if attempt_id:
                    attempt = self.attempt_storage.get_attempt(attempt_id)
                    if attempt:
                        # Show full trajectory of the attempt
                        content = f'## Full Trajectory for {attempt_id}\n\n'
                        content += f'**Phase:** {attempt.phase}\n'
                        content += f'**Summary:** {attempt.summary}\n'
                        if attempt.characterization_summary:
                            content += f'**Characterization:** {attempt.characterization_summary}\n'
                        content += f'\n**Event Range:** {attempt.start_event_id} to {attempt.end_event_id or "ongoing"}\n'
                        content += f'**Total Events:** {len(attempt.events)}\n\n'

                        # Show trajectory of events
                        if attempt.events:
                            content += '### Trajectory:\n\n'
                            for i, evt in enumerate(attempt.events, 1):
                                content += f'{i}. '
                                # Format event based on type
                                if hasattr(evt, 'command'):
                                    content += f'Command: `{evt.command}`\n'
                                elif hasattr(evt, 'path') and hasattr(evt, 'content'):
                                    content += f'File Edit: {evt.path}\n'
                                elif hasattr(evt, 'message'):
                                    content += f'Message: {evt.message[:100]}...\n' if len(evt.message) > 100 else f'Message: {evt.message}\n'
                                elif hasattr(evt, 'content'):
                                    content += f'Observation: {str(evt.content)[:100]}...\n' if len(str(evt.content)) > 100 else f'Observation: {str(evt.content)}\n'
                                else:
                                    content += f'{type(evt).__name__}\n'
                        else:
                            content += 'No events recorded for this attempt.\n'

                        event.content = content
                    else:
                        event.content = f'Attempt {attempt_id} not found.'

    def _enrich_attempt_actions(self, actions: list[Action]) -> None:
        """Enrich ExpandPreviousAttemptAction (now browse_attempt) with attempt data."""
        for action in actions:
            if isinstance(action, ExpandPreviousAttemptAction):
                attempt = self.attempt_storage.get_attempt(action.attempt_id)
                if attempt:
                    action.thought = f'Browsing full trajectory for {action.attempt_id}...'
                else:
                    action.thought = f'Attempt {action.attempt_id} not found.'

    def step(self, state: State) -> Action:
        """Perform one step using the RLM Agent.

        Args:
            state: Current state containing event history

        Returns:
            Action to execute
        """
        # Process pending actions first
        if self.pending_actions:
            return self.pending_actions.popleft()

        # Check if we should finish (after REFLECT phase)
        if self._should_finish() and self.current_phase == Phase.REFLECT:
            return self._handle_best_attempt_finish(state)

        # Handle /exit command
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # Phase-specific event handling
        if self.current_phase == Phase.ATTEMPT:
            # Handle finish_attempt
            for event in reversed(state.history[-10:]):
                if isinstance(event, FinishAttemptAction):
                    result = self._handle_finish_attempt(state, event)
                    if result:
                        return result

            # Start new attempt if needed
            if self.attempt_storage.current_attempt is None:
                start_event_id = state.history[-1].id + 1 if state.history else 0
                self.attempt_storage.start_attempt(phase=Phase.ATTEMPT, start_event_id=start_event_id)

        elif self.current_phase == Phase.CHARACTERIZE:
            # Guardrail: do not proceed unless an attempt has been finished.
            if not self._has_completed_attempt():
                self._reset_phase(Phase.ATTEMPT)
                return self._create_message_action(
                    content='Cannot characterize without a finished attempt. Call finish_attempt with a summary first.',
                    wait_for_response=True,
                    source=EventSource.USER,
                )
            # Handle finish_characterization
            for event in reversed(state.history[-10:]):
                if isinstance(event, FinishCharacterizationAction):
                    return self._handle_finish_characterization(state, event)

        else:  # Phase.REFLECT
            # Guardrail: require a completed characterization summary before reflecting.
            if not self._has_characterized_attempt():
                self._reset_phase(Phase.CHARACTERIZE)
                return self._create_message_action(
                    content='Cannot reflect before finish_characterization has been completed. Provide a characterization_summary first.',
                    wait_for_response=True,
                    source=EventSource.USER,
                )
            # Handle submit_attempt_as_final
            for event in reversed(state.history[-10:]):
                if isinstance(event, SubmitAttemptAsFinalAction):
                    return self._handle_submit_attempt_as_final(state, event)

            # Handle finish_reflection
            for event in reversed(state.history[-10:]):
                if isinstance(event, FinishReflectionAction):
                    return self._handle_finish_reflection(state, event)

        # Update current attempt events
        if self.attempt_storage.current_attempt:
            for event in state.history[-10:]:
                if event.id >= self.attempt_storage.current_attempt.start_event_id:
                    if event not in self.attempt_storage.current_attempt.events:
                        self.attempt_storage.current_attempt.events.append(event)

        # Build messages for LLM
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                return condensation_action

        # Enrich observations with attempt data
        self._enrich_attempt_observations(condensed_history, state)

        # Build messages based on phase
        if self.current_phase == Phase.REFLECT:
            messages = self._build_reflect_context(state)
        elif self.current_phase == Phase.CHARACTERIZE:
            messages = self._build_characterize_context(state)
        else:  # Phase.ATTEMPT
            # Get initial user message (only needed for ATTEMPT phase)
            initial_user_message = self._get_initial_user_message(state.history)
            messages = self.conversation_memory.process_events(
                condensed_history=condensed_history,
                initial_user_action=initial_user_message,
                max_message_chars=self.llm.config.max_message_chars,
                vision_is_active=self.llm.vision_is_active(),
            )

            if self.llm.is_caching_prompt_active():
                self.conversation_memory.apply_prompt_caching(messages)

        # Call LLM
        params: dict = {
            'messages': messages,
            'tools': check_tools(self.tools, self.llm.config),
            'extra_body': {
                'metadata': state.to_llm_metadata(
                    model_name=self.llm.config.model, agent_name=self.name
                )
            },
                'log_metadata': {
                'phase': self.current_phase.value,
                'iteration': self.current_iteration,
            },
        }
        # Encourage any model to return tool calls instead of empty messages.
        params['tool_choice'] = 'auto'
        response = self.llm.completion(**params)
        logger.debug(f'Response from LLM: {response}')

        # Convert response to actions
        allowed_tools = {tool['function']['name'] for tool in self.tools}
        actions = rlm_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
            allowed_tools=allowed_tools,
        )

        # Log phase data for debugging
        self._log_phase(messages, response, actions)

        # Enrich actions with attempt data
        self._enrich_attempt_actions(actions)

        # Guardrail: force tool call when the assistant returns plain text/empty.
        if self._should_force_tool_call(actions):
            key = self._phase_retry_key()
            if key in self._no_tool_prompted:
                # Already nudged once for this phase/iteration; skip duplicate prompt.
                pass
            else:
                corrective = self._create_message_action(
                    content=self._get_tool_call_nudge(),
                    wait_for_response=True,
                    source=EventSource.USER,
                )
                self._no_tool_prompted.add(key)
                self.pending_actions.append(corrective)
                return self.pending_actions.popleft()

        # Queue actions
        for action in actions:
            self.pending_actions.append(action)

        return self.pending_actions.popleft()

    def _log_phase(
        self,
        messages: list[Message],
        response: 'ModelResponse',
        actions: list[Action],
    ) -> None:
        """Log phase data to JSON file for debugging.

        Writes messages, tools, LLM response, and actions to a JSON file
        in the phase_logs directory, organized by iteration and phase.

        Args:
            messages: Messages sent to LLM
            response: Raw LLM response
            actions: Parsed actions from response
        """
        if not self.phase_log_dir:
            return

        try:
            # Create directory structure: phase_logs/iteration_N/
            iteration_dir = os.path.join(
                self.phase_log_dir,
                f'iteration_{self.current_iteration}'
            )
            if not os.path.exists(iteration_dir):
                os.makedirs(iteration_dir, exist_ok=True)
                logger.info(f'Created phase log directory: {iteration_dir}')

            # Track step count per phase key
            phase_key = f'{self.current_iteration}_{self.current_phase.value}'
            step_num = self._phase_step_counter.get(phase_key, 0)
            self._phase_step_counter[phase_key] = step_num + 1

            # Filename: PHASE_step_N.json
            filename = f'{self.current_phase.value}_step_{step_num}.json'
            filepath = os.path.join(iteration_dir, filename)

            # Serialize messages
            serialized_messages = []
            for msg in messages:
                msg_dict: dict[str, Any] = {'role': msg.role, 'content': []}
                for content_item in msg.content:
                    if hasattr(content_item, 'text'):
                        msg_dict['content'].append({'type': 'text', 'text': content_item.text})
                    elif hasattr(content_item, 'name'):
                        # Tool result content
                        msg_dict['content'].append({
                            'type': 'tool_result',
                            'tool_call_id': getattr(content_item, 'tool_call_id', None),
                            'name': getattr(content_item, 'name', None),
                            'output': getattr(content_item, 'output', None),
                        })
                    elif hasattr(content_item, 'image_urls'):
                        msg_dict['content'].append({'type': 'image', 'urls': content_item.image_urls})
                    else:
                        msg_dict['content'].append({'type': 'unknown', 'data': str(content_item)})
                serialized_messages.append(msg_dict)

            # Serialize tools (just names and descriptions for readability)
            serialized_tools = []
            for tool in self.tools:
                tool_info = {
                    'name': tool.get('function', {}).get('name', 'unknown'),
                    'description': tool.get('function', {}).get('description', '')[:200],  # Truncate
                }
                serialized_tools.append(tool_info)

            # Serialize LLM response
            serialized_response: dict[str, Any] = {}
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message'):
                    msg = choice.message
                    serialized_response['content'] = getattr(msg, 'content', None)
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        serialized_response['tool_calls'] = [
                            {
                                'id': tc.id,
                                'function': {
                                    'name': tc.function.name,
                                    'arguments': tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ]
            if hasattr(response, 'usage'):
                serialized_response['usage'] = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0),
                }

            # Serialize actions
            serialized_actions = []
            for action in actions:
                action_dict: dict[str, Any] = {
                    'type': type(action).__name__,
                }
                # Include common action attributes
                if hasattr(action, 'thought'):
                    action_dict['thought'] = action.thought
                if hasattr(action, 'command'):
                    action_dict['command'] = action.command
                if hasattr(action, 'message'):
                    action_dict['message'] = action.message
                if hasattr(action, 'path'):
                    action_dict['path'] = action.path
                if hasattr(action, 'content'):
                    action_dict['content'] = str(action.content)[:500] if action.content else None  # Truncate
                serialized_actions.append(action_dict)

            # Build log entry
            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'phase': self.current_phase.value,
                'iteration': self.current_iteration,
                'step': step_num,
                'messages': serialized_messages,
                'tools_available': serialized_tools,
                'llm_response': serialized_response,
                'actions': serialized_actions,
            }

            # Write to file with explicit flush for real-time visibility
            with open(filepath, 'w') as f:
                json.dump(log_entry, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk

            logger.info(f'Phase log written: {filename} (phase={self.current_phase.value}, iteration={self.current_iteration}, step={step_num})')

        except Exception as e:
            logger.warning(f'Failed to write phase log: {e}')

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Find the initial user message action from history.

        For subsequent ATTEMPT iterations, prefer the most recent user-sourced
        transition/insights message (e.g., the reflection handoff) so the new
        attempt starts with the latest plan. For the first iteration, fall back
        to the very first user request.
        """
        def _is_user_message(event: Event) -> bool:
            return isinstance(event, MessageAction) and getattr(event, 'source', None) in (
                EventSource.USER,
                'user',
            )

        if self.current_phase == Phase.ATTEMPT and self.current_iteration > 0:
            for event in reversed(history):
                if _is_user_message(event):
                    return event

        for event in history:
            if _is_user_message(event):
                return event

        logger.error(f'Could not find user MessageAction in {len(history)} events')
        raise ValueError('User message not found in history. Please report this issue.')

