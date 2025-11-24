import os
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

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
from openhands.agenthub.codeact_agent.tools.finish import FinishTool
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
from openhands.agenthub.rlm_agent.tools.browse_previous_attempts import (
    BrowsePreviousAttemptsTool,
)
from openhands.agenthub.rlm_agent.tools.expand_previous_attempt import (
    ExpandPreviousAttemptTool,
)
from openhands.agenthub.rlm_agent.tools.finish_browsing_attempts import (
    FinishBrowsingAttemptTool,
)
from openhands.agenthub.rlm_agent.tools.submit_attempt_as_final import (
    SubmitAttemptAsFinalTool,
)
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.events.action import (
    Action,
    AgentFinishAction,
    BrowsePreviousAttemptsAction,
    CmdRunAction,
    ExpandPreviousAttemptAction,
    FinishAttemptAction,
    FinishBrowsingAttemptAction,
    MessageAction,
    RecallAction,
    SubmitAttemptAsFinalAction,
)
from openhands.events.action.agent import CondensationAction
from openhands.events.observation import (
    BrowsePreviousAttemptsObservation,
    CmdOutputObservation,
    ExpandPreviousAttemptObservation,
    FinishAttemptObservation,
    SubmitAttemptAsFinalObservation,
)
from openhands.events.event import Event
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
    RLM = 'RLM'


class BrowsingState(str, Enum):
    """Represents the browsing state during an attempt."""

    NONE = 'NONE'  # Not browsing
    BROWSING = 'BROWSING'  # Currently browsing previous attempts


class RLMAgent(Agent):
    VERSION = '1.0'
    """
    The RLM (Recursive Language Model) Agent alternates between two phases:

    1. ATTEMPT phase: The agent tries to solve the task directly, similar to CodeAct agent.
       When done, it calls finish_attempt to store the attempt.

    2. RLM phase: The agent reviews previous attempts, expands on them, reasons about them,
       and creates a plan for the next attempt. When done, it calls finish_browsing_attempt.

    3. During ATTEMPT phase the agent can also browse previous attempts to get insights and plan the next action.
       The agent calls browse_previous_attempts to browse previous attempts and expand_previous_attempt to get more details about the attempts.
       The agent can then call finish_browsing_attempt to finish the browsing session and return to the ATTEMPT phase.
       The finish_browsing_attempt action collapses the browsing session into a single summary to inform next action.

    The agent alternates between these phases for a predefined number of iterations,
    then returns the best solution found.
    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        """Initializes a new instance of the RLMAgent class.

        Parameters:
        - config (AgentConfig): The configuration for this agent
        """
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.reset()
        self.tools = self._get_tools()

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        # Override with router if needed
        self.llm = self.llm_registry.get_router(self.config)

        # RLM-specific state
        self.attempt_storage = AttemptStorage()
        self.current_phase: Phase = Phase.ATTEMPT
        self.browsing_state: BrowsingState = BrowsingState.NONE
        self.browsing_start_event_id: int | None = None  # Track where browsing started
        # Get rlm_max_iterations from extended config or environment variable, defaulting to 3
        rlm_max_iterations = None
        if hasattr(config, 'extended') and config.extended:
            try:
                rlm_max_iterations = config.extended.get('rlm_max_iterations')
            except (KeyError, AttributeError):
                pass
        if rlm_max_iterations is None:
            rlm_max_iterations = int(os.environ.get('RLM_MAX_ITERATIONS', '3'))
        self.max_iterations: int = rlm_max_iterations
        self.current_iteration: int = 0

        # Get patch extraction and application commands from extended config
        # These will be read dynamically from config.extended when needed
        self._extract_patch_cmd: str | None = None
        self._apply_patch_cmd: str | None = None

        # Track state for patch extraction
        self._pending_patch_extraction: tuple[int, int] | None = None  # (finish_attempt_event_id, extract_cmd_action_id)
        self._pending_patch_application: bool = False
        self._apply_patch_action_id: int | None = None

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
                system_prompt_filename=self._get_system_prompt_filename(),
            )

        return self._prompt_manager

    def _get_system_prompt_filename(self) -> str:
        """Get the system prompt filename based on current phase."""
        if self.current_phase == Phase.ATTEMPT:
            return 'system_prompt_attempt.j2'
        else:
            return 'system_prompt_rlm.j2'

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        # TODO: Don't use short description for cmd run tool.
        use_short_tool_desc = False

        tools = []

        # Phase-specific tools
        if self.current_phase == Phase.ATTEMPT:
            if self.browsing_state == BrowsingState.BROWSING:
                # During browsing, only allow browsing-related tools
                tools.append(BrowsePreviousAttemptsTool)
                tools.append(ExpandPreviousAttemptTool)
                tools.append(FinishBrowsingAttemptTool)
                tools.append(SubmitAttemptAsFinalTool)  # Can submit attempt as final during browsing
                if self.config.enable_think:
                    tools.append(ThinkTool)
            else:
                # In ATTEMPT phase, include all standard tools plus finish_attempt and browse_previous_attempts
                if self.config.enable_cmd:

                    tools.append(create_cmd_run_tool(use_short_description=False))
                if self.config.enable_think:
                    tools.append(ThinkTool)
                if self.config.enable_browsing:
                    import sys

                    if sys.platform == 'win32':
                        logger.warning('Windows runtime does not support browsing yet')
                    else:
                        tools.append(BrowserTool)
                if self.config.enable_jupyter:
                    tools.append(IPythonTool)
                if self.config.enable_plan_mode:
                    tools.append(create_task_tracker_tool(use_short_tool_desc))
                if self.config.enable_llm_editor:
                    tools.append(LLMBasedFileEditTool)
                elif self.config.enable_editor:
                    tools.append(
                        create_str_replace_editor_tool(
                            use_short_description=use_short_tool_desc,
                            runtime_type=self.config.runtime,
                        )
                    )
                if self.config.enable_condensation_request:
                    tools.append(CondensationRequestTool)
                # RLM-specific tools for ATTEMPT phase
                tools.append(FinishAttemptTool)
                tools.append(BrowsePreviousAttemptsTool)  # Can browse during attempt
        else:
            # In RLM phase, only include RLM-specific tools
            tools.append(BrowsePreviousAttemptsTool)
            tools.append(ExpandPreviousAttemptTool)
            tools.append(FinishBrowsingAttemptTool)
            tools.append(SubmitAttemptAsFinalTool)  # Can submit attempt as final in RLM phase
            # Also allow thinking
            if self.config.enable_think:
                tools.append(ThinkTool)

        return tools

    def reset(self) -> None:
        """Resets the RLM Agent's internal state."""
        super().reset()
        # Only clear pending actions, not LLM metrics
        self.pending_actions.clear()
        # Reset RLM-specific state
        self.attempt_storage = AttemptStorage()
        self.current_phase = Phase.ATTEMPT
        self.browsing_state = BrowsingState.NONE
        self.browsing_start_event_id = None
        self.current_iteration = 0
        self._pending_patch_extraction = None
        self._pending_patch_application = False
        self._apply_patch_action_id = None
        self._pending_patch_extraction = None
        self._pending_patch_application = False

    def _transition_to_next_phase(self) -> None:
        """Transition to the next phase."""
        if self.current_phase == Phase.ATTEMPT:
            self.current_phase = Phase.RLM
            logger.info('Transitioning to RLM phase')
        else:
            self.current_phase = Phase.ATTEMPT
            self.current_iteration += 1
            logger.info(f'Transitioning to ATTEMPT phase (iteration {self.current_iteration})')
        # Update tools for new phase
        self.tools = self._get_tools()
        # Reset prompt manager to use new system prompt
        self._prompt_manager = None

    def _should_finish(self) -> bool:
        """Check if we should finish and return the best solution."""
        return self.current_iteration >= self.max_iterations

    def _collapse_browsing_session(self, state: State) -> CondensationAction | None:
        """Collapse the browsing session into a single summary to avoid context rot.

        This method identifies all events between browsing_start_event_id and finish_browsing_attempt,
        and creates a condensation action to collapse them.

        Returns:
            CondensationAction if there are events to collapse, None otherwise
        """
        if self.browsing_start_event_id is None:
            return None

        # Find the finish_browsing_attempt event
        finish_event_id = None
        finish_message = ''
        for event in reversed(state.history[-20:]):
            if isinstance(event, FinishBrowsingAttemptAction):
                finish_event_id = event.id
                finish_message = event.message
                break

        if finish_event_id is None:
            return None

        # Find all events in the browsing session (excluding start and finish actions)
        browsing_event_ids = []
        for event in state.history:
            if (event.id > self.browsing_start_event_id and
                event.id < finish_event_id):
                browsing_event_ids.append(event.id)

        if browsing_event_ids:
            logger.info(f'Collapsing {len(browsing_event_ids)} events from browsing session')
            # Create a condensation action to collapse the browsing session
            # Use range if events are contiguous, otherwise use list
            if (len(browsing_event_ids) > 1 and
                browsing_event_ids[-1] - browsing_event_ids[0] == len(browsing_event_ids) - 1):
                return CondensationAction(
                    forgotten_events_start_id=browsing_event_ids[0],
                    forgotten_events_end_id=browsing_event_ids[-1],
                    summary=f'Browsing session: {finish_message}',
                    summary_offset=self.browsing_start_event_id,
                )
            else:
                return CondensationAction(
                    forgotten_event_ids=browsing_event_ids,
                    summary=f'Browsing session: {finish_message}',
                    summary_offset=self.browsing_start_event_id,
                )

        return None

    def _create_message_action(
        self, content: str, wait_for_response: bool = False
    ) -> MessageAction:
        """Create and validate a MessageAction to ensure content is never None.

        This prevents errors when the controller creates RecallAction from MessageAction.
        """
        # Ensure content is always a valid string (not None)
        if content is None:
            logger.warning('MessageAction content is None, using empty string')
            content = ''
        elif not isinstance(content, str):
            content = str(content)
        return MessageAction(content=content, wait_for_response=wait_for_response)

    def _validate_action(self, action: 'Action') -> 'Action':
        """Validate and fix action to prevent serialization errors.

        Ensures MessageAction has valid content and RecallAction has valid query.
        This prevents errors when the controller creates RecallAction from MessageAction.
        """
        if isinstance(action, MessageAction):
            # Ensure content is always a valid string (not None)
            # This prevents errors when controller creates RecallAction(query=action.content)
            if action.content is None:
                logger.warning(f'MessageAction with None content detected, fixing to empty string')
                action.content = ''
            elif not isinstance(action.content, str):
                action.content = str(action.content)
        elif isinstance(action, RecallAction):
            # Ensure query is always a valid string (not None)
            # This prevents errors when serializing RecallAction.message property
            if action.query is None:
                logger.warning(f'RecallAction with None query detected, fixing to empty string')
                action.query = ''
            elif not isinstance(action.query, str):
                action.query = str(action.query)
        return action

    def step(self, state: State) -> 'Action':
        """Performs one step using the RLM Agent.

        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - Various actions depending on the phase and what the model decides to do
        """
        # Continue with pending actions if any
        if self.pending_actions:
            action = self.pending_actions.popleft()
            return self._validate_action(action)

        # Check if we should finish
        if self._should_finish() and self.current_phase == Phase.RLM:
            # Check if we need to apply the best attempt's patch
            if self._pending_patch_application:
                # Check if patch application is complete
                for event in reversed(state.history[-5:]):
                    if (
                        isinstance(event, CmdOutputObservation)
                        and hasattr(self, '_apply_patch_action_id')
                        and event.cause == self._apply_patch_action_id
                    ):
                        # Patch application completed
                        self._pending_patch_application = False
                        prompt_dir = os.path.join(
                            os.path.dirname(__file__), 'prompts'
                        )
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
                # Still waiting for patch application
                return self._create_message_action(
                    content='Applying best attempt patch...',
                    wait_for_response=False,
                )

            # Apply best attempt's patch if command is configured
            prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
            best_attempt = self.attempt_storage.get_best_attempt(
                llm=self.llm, prompt_dir=prompt_dir
            )
            # Read apply command from config (may have been set after agent initialization)
            apply_cmd = None
            if hasattr(self.config, 'extended') and self.config.extended:
                extended_dict = self.config.extended.model_dump()
                apply_cmd = extended_dict.get('rlm_apply_patch_cmd')
            if best_attempt and best_attempt.patch and apply_cmd:
                # Apply the patch - use base64 encoding to avoid shell escaping issues
                import base64
                patch_b64 = base64.b64encode(best_attempt.patch.encode()).decode()
                apply_cmd_final = apply_cmd.replace('{patch}', patch_b64)
                apply_action = CmdRunAction(command=apply_cmd_final)
                apply_action.set_hard_timeout(600)
                self.pending_actions.append(apply_action)
                self._pending_patch_application = True
                self._apply_patch_action_id = apply_action.id
                logger.info(
                    f'Applying patch from best attempt {best_attempt.attempt_id}'
                )
                return apply_action
            elif best_attempt:
                return AgentFinishAction(
                    final_thought=f'Completed {self.current_iteration} iterations. Best attempt: {best_attempt.attempt_id}. {best_attempt.summary}'
                )
            else:
                return AgentFinishAction(
                    final_thought=f'Completed {self.current_iteration} iterations. No completed attempts found.'
                )

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if (
            latest_user_message
            and latest_user_message.content
            and latest_user_message.content.strip() == '/exit'
        ):
            return AgentFinishAction()

        # Handle phase-specific actions and browsing state
        if self.current_phase == Phase.ATTEMPT:
            # Check if we just started browsing (look for the observation, which means action was executed)
            for event in reversed(state.history[-10:]):  # Check last 10 events
                if isinstance(event, BrowsePreviousAttemptsObservation):
                    # Find the corresponding action to get its ID
                    for action_event in reversed(state.history):
                        if isinstance(action_event, BrowsePreviousAttemptsAction) and action_event.id == event.cause:
                            if self.browsing_state == BrowsingState.NONE:
                                self.browsing_state = BrowsingState.BROWSING
                                self.browsing_start_event_id = action_event.id
                                logger.info('Started browsing previous attempts during attempt')
                                # Update tools to only allow browsing
                                self.tools = self._get_tools()
                                break
                            break
                    break

            # Check if we finished browsing
            if self.browsing_state == BrowsingState.BROWSING:
                # Check if agent submitted an attempt as final during browsing
                for event in reversed(state.history[-5:]):  # Check last 5 events
                    if isinstance(event, SubmitAttemptAsFinalAction):
                        # Get the attempt to submit
                        attempt = self.attempt_storage.get_attempt(event.attempt_id)
                        if attempt:
                            logger.info(f'Submitting attempt {event.attempt_id} as final solution')
                            # Check if we need to apply the attempt's patch
                            if self._pending_patch_application:
                                # Check if patch application is complete
                                for obs_event in reversed(state.history[-5:]):
                                    if (
                                        isinstance(obs_event, CmdOutputObservation)
                                        and hasattr(self, '_apply_patch_action_id')
                                        and obs_event.cause == self._apply_patch_action_id
                                    ):
                                        # Patch application completed
                                        self._pending_patch_application = False
                                        return AgentFinishAction(
                                            final_thought=f'Submitted attempt {event.attempt_id} as final solution. {event.message}. Applied patch successfully.'
                                        )
                                # Still waiting for patch application
                                return self._create_message_action(
                                    content='Applying patch for submitted attempt...',
                                    wait_for_response=False,
                                )

                            # Apply attempt's patch if command is configured
                            apply_cmd = None
                            if hasattr(self.config, 'extended') and self.config.extended:
                                extended_dict = self.config.extended.model_dump()
                                apply_cmd = extended_dict.get('rlm_apply_patch_cmd')
                            if attempt.patch and apply_cmd:
                                # Apply the patch - use base64 encoding to avoid shell escaping issues
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
                                # No patch to apply, finish immediately
                                return AgentFinishAction(
                                    final_thought=f'Submitted attempt {event.attempt_id} as final solution. {event.message}'
                                )
                        else:
                            logger.warning(f'Attempt {event.attempt_id} not found, cannot submit as final')
                            return self._create_message_action(
                                content=f'Attempt {event.attempt_id} not found. Cannot submit as final.',
                                wait_for_response=False,
                            )

                for event in reversed(state.history[-5:]):  # Check last 5 events
                    if isinstance(event, FinishBrowsingAttemptAction):
                        # Collapse browsing session
                        condensation_action = self._collapse_browsing_session(state)
                        self.browsing_state = BrowsingState.NONE
                        self.browsing_start_event_id = None
                        logger.info('Finished browsing previous attempts, resuming attempt')
                        # Update tools to allow all attempt tools again
                        self.tools = self._get_tools()

                        # Return condensation action if we have events to collapse
                        if condensation_action:
                            return condensation_action
                        else:
                            # If no events to collapse, just return a message
                            insights = event.message if event.message else 'No specific insights'
                            return self._create_message_action(
                                content=f'Browsing session completed. Insights: {insights}. Resuming attempt.',
                                wait_for_response=False,
                            )

            # Check if we just finished an attempt
            for event in reversed(state.history[-10:]):  # Check last 10 events
                if isinstance(event, FinishAttemptAction):
                    # Check if we've already processed this finish_attempt
                    if self._pending_patch_extraction is not None:
                        if self._pending_patch_extraction[0] == event.id:
                            # We're waiting for patch extraction, check if it's done
                            for obs_event in reversed(state.history[-5:]):
                                if (
                                    isinstance(obs_event, CmdOutputObservation)
                                    and obs_event.cause == self._pending_patch_extraction[1]
                                ):
                                    # Patch extraction completed, store it
                                    patch = obs_event.content
                                    if self.attempt_storage.current_attempt:
                                        self.attempt_storage.finish_attempt(
                                            end_event_id=event.id,
                                            summary=event.message,
                                            patch=patch,
                                        )
                                    self._pending_patch_extraction = None
                                    # Transition to RLM phase for reflection
                                    self._transition_to_next_phase()
                                    return self._create_message_action(
                                        content=f'Attempt completed. Transitioning to RLM reflection phase to review and plan next attempt.',
                                        wait_for_response=False,
                                    )
                            # Still waiting for patch extraction
                            return self._create_message_action(
                                content='Extracting patch for attempt...',
                                wait_for_response=False,
                            )

                    # First time seeing this finish_attempt, extract patch if command is configured
                    # Read command from config (may have been set after agent initialization)
                    extract_cmd = None
                    if hasattr(self.config, 'extended') and self.config.extended:
                        extended_dict = self.config.extended.model_dump()
                        extract_cmd = extended_dict.get('rlm_extract_patch_cmd')
                    if extract_cmd and self.attempt_storage.current_attempt:
                        # Queue patch extraction command
                        extract_action = CmdRunAction(command=extract_cmd)
                        extract_action.set_hard_timeout(600)
                        self.pending_actions.append(extract_action)
                        self._pending_patch_extraction = (event.id, extract_action.id)
                        logger.info(
                            f'Extracting patch for attempt {self.attempt_storage.current_attempt.attempt_id}'
                        )
                        return extract_action
                    else:
                        # No patch extraction command configured, finish normally
                        if self.attempt_storage.current_attempt:
                            self.attempt_storage.finish_attempt(
                                end_event_id=event.id, summary=event.message
                            )
                        # Transition to RLM phase for reflection
                        self._transition_to_next_phase()
                        return self._create_message_action(
                            content=f'Attempt completed. Transitioning to RLM reflection phase to review and plan next attempt.',
                            wait_for_response=False,
                        )
        else:
            # RLM phase - check if agent submitted an attempt as final
            for event in reversed(state.history[-10:]):  # Check last 10 events
                if isinstance(event, SubmitAttemptAsFinalAction):
                    # Get the attempt to submit
                    attempt = self.attempt_storage.get_attempt(event.attempt_id)
                    if attempt:
                        logger.info(f'Submitting attempt {event.attempt_id} as final solution')
                        # Check if we need to apply the attempt's patch
                        if self._pending_patch_application:
                            # Check if patch application is complete
                            for obs_event in reversed(state.history[-5:]):
                                if (
                                    isinstance(obs_event, CmdOutputObservation)
                                    and hasattr(self, '_apply_patch_action_id')
                                    and obs_event.cause == self._apply_patch_action_id
                                ):
                                    # Patch application completed
                                    self._pending_patch_application = False
                                    return AgentFinishAction(
                                        final_thought=f'Submitted attempt {event.attempt_id} as final solution. {event.message}. Applied patch successfully.'
                                    )
                            # Still waiting for patch application
                            return self._create_message_action(
                                content='Applying patch for submitted attempt...',
                                wait_for_response=False,
                            )

                        # Apply attempt's patch if command is configured
                        apply_cmd = None
                        if hasattr(self.config, 'extended') and self.config.extended:
                            extended_dict = self.config.extended.model_dump()
                            apply_cmd = extended_dict.get('rlm_apply_patch_cmd')
                        if attempt.patch and apply_cmd:
                            # Apply the patch - use base64 encoding to avoid shell escaping issues
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
                            # No patch to apply, finish immediately
                            return AgentFinishAction(
                                final_thought=f'Submitted attempt {event.attempt_id} as final solution. {event.message}'
                            )
                    else:
                        logger.warning(f'Attempt {event.attempt_id} not found, cannot submit as final')
                        return self._create_message_action(
                            content=f'Attempt {event.attempt_id} not found. Cannot submit as final.',
                            wait_for_response=False,
                        )

            # RLM phase - check if we finished browsing/reflection
            for event in reversed(state.history[-10:]):  # Check last 10 events
                if isinstance(event, FinishBrowsingAttemptAction):
                    # Transition to ATTEMPT phase
                    self._transition_to_next_phase()
                    # Start a new attempt
                    if state.history:
                        start_event_id = state.history[-1].id + 1
                    else:
                        start_event_id = 0
                    self.attempt_storage.start_attempt(
                        phase=Phase.ATTEMPT, start_event_id=start_event_id
                    )
                    # Add a message about transitioning
                    return self._create_message_action(
                        content=f'RLM reflection phase completed. Starting new attempt phase (iteration {self.current_iteration}).',
                        wait_for_response=False,
                    )

        # Start a new attempt if we're in ATTEMPT phase and don't have one
        if self.current_phase == Phase.ATTEMPT and self.attempt_storage.current_attempt is None:
            if state.history:
                start_event_id = state.history[-1].id + 1
            else:
                start_event_id = 0
            self.attempt_storage.start_attempt(
                phase=Phase.ATTEMPT, start_event_id=start_event_id
            )

        # Validate all MessageAction objects in state history to prevent serialization errors
        # This ensures that when the controller processes these events, they have valid content
        for event in state.history:
            if isinstance(event, MessageAction):
                self._validate_action(event)

        # Track events in current attempt
        if self.attempt_storage.current_attempt:
            # Add recent events to the current attempt
            for event in state.history[-10:]:
                if event.id >= self.attempt_storage.current_attempt.start_event_id:
                    if event not in self.attempt_storage.current_attempt.events:
                        self.attempt_storage.current_attempt.events.append(event)

        # Condense the events from the state. If we get a view we'll pass those
        # to the conversation manager for processing, but if we get a condensation
        # event we'll just return that instead of an action. The controller will
        # immediately ask the agent to step again with the new view.
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        # Validate all MessageAction objects in condensed_history to prevent serialization errors
        # This ensures that when the controller processes these events, they have valid content
        for event in condensed_history:
            if isinstance(event, MessageAction):
                self._validate_action(event)

        # Populate RLM observations with attempt data
        for event in condensed_history:
            if isinstance(event, BrowsePreviousAttemptsObservation):
                summaries = self.attempt_storage.get_summarized_attempts()
                if summaries:
                    content = 'Previous attempts:\n'
                    for s in summaries:
                        content += f"- {s['id']} ({s['phase']}): {s['summary']}\n"
                    event.content = content
                else:
                    event.content = 'No previous attempts found.'
            elif isinstance(event, ExpandPreviousAttemptObservation):
                # Find the corresponding ExpandPreviousAttemptAction in recent events
                # to get the attempt_id
                attempt_id = None
                for prev_event in reversed(state.history):
                    if isinstance(prev_event, ExpandPreviousAttemptAction):
                        if prev_event.id == event.cause:
                            attempt_id = prev_event.attempt_id
                            break

                if attempt_id:
                    attempt = self.attempt_storage.get_attempt(attempt_id)
                    if attempt:
                        content = f'Expanded details for {attempt_id}:\n'
                        content += f'Phase: {attempt.phase}\n'
                        content += f'Summary: {attempt.summary}\n'
                        content += f'Events: {len(attempt.events)} events\n'
                        if attempt.end_event_id:
                            content += f'Event IDs: {attempt.start_event_id} to {attempt.end_event_id}\n'
                        event.content = content
                    else:
                        event.content = f'Attempt {attempt_id} not found.'
                else:
                    # Fallback: try to extract from content
                    import re
                    match = re.search(r'attempt-(\d+)', event.content)
                    if match:
                        attempt_id = f'attempt-{match.group(1)}'
                        attempt = self.attempt_storage.get_attempt(attempt_id)
                        if attempt:
                            content = f'Expanded details for {attempt_id}:\n'
                            content += f'Phase: {attempt.phase}\n'
                            content += f'Summary: {attempt.summary}\n'
                            content += f'Events: {len(attempt.events)} events\n'
                            if attempt.end_event_id:
                                content += f'Event IDs: {attempt.start_event_id} to {attempt.end_event_id}\n'
                            event.content = content

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message, state)

        params: dict = {
            'messages': messages,
        }
        params['tools'] = check_tools(self.tools, self.llm.config)
        params['extra_body'] = {
            'metadata': state.to_llm_metadata(
                model_name=self.llm.config.model, agent_name=self.name
            )
        }
        response = self.llm.completion(**params)
        logger.debug(f'Response from LLM: {response}')
        actions = self.response_to_actions(response)
        logger.debug(f'Actions after response_to_actions: {actions}')

        # Validate all actions to prevent serialization errors
        # This ensures MessageAction has valid content (prevents RecallAction creation errors in controller)
        # and RecallAction has valid query
        for action in actions:
            self._validate_action(action)

        # Handle RLM-specific actions
        for action in actions:
            if isinstance(action, BrowsePreviousAttemptsAction):
                # Return summarized attempts
                summaries = self.attempt_storage.get_summarized_attempts()
                # This will be handled by creating an observation
                # For now, we'll add it as a thought
                action.thought = f'Previous attempts:\n' + '\n'.join(
                    [
                        f"- {s['id']} ({s['phase']}): {s['summary']}"
                        for s in summaries
                    ]
                )
            elif isinstance(action, ExpandPreviousAttemptAction):
                attempt = self.attempt_storage.get_attempt(action.attempt_id)
                if attempt:
                    action.thought = f'Expanded details for {action.attempt_id}:\n'
                    action.thought += f'Phase: {attempt.phase}\n'
                    action.thought += f'Summary: {attempt.summary}\n'
                    action.thought += f'Events: {len(attempt.events)} events\n'
                else:
                    action.thought = f'Attempt {action.attempt_id} not found.'

        for action in actions:
            self._validate_action(action)
            self.pending_actions.append(action)

        action = self.pending_actions.popleft()
        return self._validate_action(action)

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        """Finds the initial user message action from the full history."""
        initial_user_message: MessageAction | None = None
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                initial_user_message = event
                break

        if initial_user_message is None:
            # This should not happen in a valid conversation
            logger.error(
                f'CRITICAL: Could not find the initial user MessageAction in the full {len(history)} events history.'
            )
            # Depending on desired robustness, could raise error or create a dummy action
            # and log the error
            raise ValueError(
                'Initial user message not found in history. Please report this issue.'
            )
        # Validate the message action to ensure content is never None
        # This prevents errors when the controller creates RecallAction from MessageAction
        self._validate_action(initial_user_message)
        return initial_user_message

    def _get_messages(
        self,
        events: list[Event],
        initial_user_message: MessageAction,
        state: State,
    ) -> list[Message]:
        """Constructs the message history for the LLM conversation.

        This method builds a structured conversation history by processing events from the state
        and formatting them into messages that the LLM can understand.

        Args:
            events: The list of events to convert to messages
            initial_user_message: The initial user message
            state: The current state (used for RLM phase to add attempt summaries)

        Returns:
            list[Message]: A list of formatted messages ready for LLM consumption
        """
        if not self.prompt_manager:
            raise Exception('Prompt Manager not instantiated.')

        # Use ConversationMemory to process events (including SystemMessageAction)
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        # Add attempt summaries in RLM phase
        if self.current_phase == Phase.RLM:
            summaries = self.attempt_storage.get_summarized_attempts()
            if summaries:
                summary_text = 'Previous attempts summary:\n'
                for s in summaries:
                    summary_text += f"- {s['id']} ({s['phase']}): {s['summary']}\n"
                # Add as a system message or append to the last message
                # For simplicity, we'll add it as context in the system message
                # This would ideally be handled by modifying the system prompt

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return rlm_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )

