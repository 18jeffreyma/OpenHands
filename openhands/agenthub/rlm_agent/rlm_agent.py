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
from openhands.core.message import Message
from openhands.events.action import (
    Action,
    AgentFinishAction,
    BrowsePreviousAttemptsAction,
    CmdRunAction,
    ExpandPreviousAttemptAction,
    FinishAttemptAction,
    FinishCharacterizationAction,
    FinishReflectionAction,
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
    CHARACTERIZE = 'CHARACTERIZE'
    REFLECT = 'REFLECT'


class BrowsingState(str, Enum):
    """Represents the browsing state during an attempt."""

    NONE = 'NONE'  # Not browsing
    BROWSING = 'BROWSING'  # Currently browsing previous attempts


class RLMAgent(Agent):
    VERSION = '2.0'
    """
    The RLM (Recursive Language Model) Agent alternates between three phases:

    1. ATTEMPT phase: The agent tries to solve the task directly, similar to CodeAct agent.
       When done, it calls finish_attempt to store the attempt.

    2. CHARACTERIZE phase: After finishing an attempt, the agent analyzes and characterizes
       the attempt by running profiling/testing commands and creating a semantic summary.
       This summary is used to identify and compare attempts during reflection.
       When done, it calls finish_characterization with a comprehensive summary.

    3. REFLECT phase: The agent reviews previous attempts, expands on them, reasons about them,
       and creates a plan for the next attempt. When done, it calls finish_reflection.

    4. During ATTEMPT phase the agent can also browse previous attempts to get insights.
       The agent calls browse_previous_attempts to browse and expand_previous_attempt for details.
       Then calls finish_reflection to finish the browsing session and return to the ATTEMPT phase.
       The finish_reflection action collapses the browsing session into a single summary.

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
        self.browsing_start_event_id: int | None = None
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

        self._extract_patch_cmd: str | None = None
        self._apply_patch_cmd: str | None = None
        self._pending_patch_extraction: tuple[int, int] | None = None
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
        elif self.current_phase == Phase.CHARACTERIZE:
            return 'system_prompt_characterize.j2'
        else:  # Phase.REFLECT
            return 'system_prompt_reflect.j2'

    def _get_tools(self) -> list['ChatCompletionToolParam']:
        use_short_tool_desc = False

        tools = []

        # Phase-specific tools
        if self.current_phase == Phase.ATTEMPT:
            if self.browsing_state == BrowsingState.BROWSING:
                tools.append(BrowsePreviousAttemptsTool)
                tools.append(ExpandPreviousAttemptTool)
                tools.append(FinishReflectionTool)
                tools.append(SubmitAttemptAsFinalTool)
                if self.config.enable_think:
                    tools.append(ThinkTool)
            else:
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
                tools.append(FinishAttemptTool)
                tools.append(BrowsePreviousAttemptsTool)
        elif self.current_phase == Phase.CHARACTERIZE:
            if self.config.enable_cmd:
                tools.append(create_cmd_run_tool(use_short_description=False))
            if self.config.enable_think:
                tools.append(ThinkTool)
            tools.append(FinishCharacterizationTool)
        else:  # Phase.REFLECT
            tools.append(BrowsePreviousAttemptsTool)
            tools.append(ExpandPreviousAttemptTool)
            tools.append(FinishReflectionTool)
            tools.append(SubmitAttemptAsFinalTool)
            if self.config.enable_think:
                tools.append(ThinkTool)

        return tools

    def reset(self) -> None:
        """Resets the RLM Agent's internal state."""
        super().reset()
        self.pending_actions.clear()
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
        """Transition to the next phase.

        The phase flow is: ATTEMPT → CHARACTERIZE → REFLECT → ATTEMPT (new iteration)
        """
        if self.current_phase == Phase.ATTEMPT:
            self.current_phase = Phase.CHARACTERIZE
            logger.info('Transitioning to CHARACTERIZE phase')
        elif self.current_phase == Phase.CHARACTERIZE:
            self.current_phase = Phase.REFLECT
            logger.info('Transitioning to REFLECT phase')
        else:  # Phase.REFLECT
            self.current_phase = Phase.ATTEMPT
            self.current_iteration += 1
            logger.info(
                f'Transitioning to ATTEMPT phase (iteration {self.current_iteration})'
            )
        self.tools = self._get_tools()
        self._prompt_manager = None

    def _should_finish(self) -> bool:
        """Check if we should finish and return the best solution."""
        return self.current_iteration >= self.max_iterations

    def _collapse_browsing_session(self, state: State) -> CondensationAction | None:
        """Collapse the browsing session into a single summary to avoid context rot.

        This method identifies all events between browsing_start_event_id and finish_reflection,
        and creates a condensation action to collapse them.

        Returns:
            CondensationAction if there are events to collapse, None otherwise
        """
        if self.browsing_start_event_id is None:
            return None

        finish_event_id = None
        finish_message = ''
        for event in reversed(state.history[-20:]):
            if isinstance(event, FinishReflectionAction):
                finish_event_id = event.id
                finish_message = event.message
                break

        if finish_event_id is None:
            return None

        browsing_event_ids = []
        for event in state.history:
            if (event.id > self.browsing_start_event_id and
                event.id < finish_event_id):
                browsing_event_ids.append(event.id)

        if browsing_event_ids:
            logger.info(f'Collapsing {len(browsing_event_ids)} events from browsing session')
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
        if content is None:
            logger.warning('MessageAction content is None, using empty string')
            content = ''
        elif not isinstance(content, str):
            content = str(content)
        return MessageAction(content=content, wait_for_response=wait_for_response)

    def _validate_action(self, action: 'Action') -> 'Action':
        if isinstance(action, MessageAction):
            if action.content is None:
                logger.warning(f'MessageAction with None content detected, fixing to empty string')
                action.content = ''
            elif not isinstance(action.content, str):
                action.content = str(action.content)
        elif isinstance(action, RecallAction):
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
        if self.pending_actions:
            action = self.pending_actions.popleft()
            return self._validate_action(action)

        if self._should_finish() and self.current_phase == Phase.REFLECT:
            if self._pending_patch_application:
                for event in reversed(state.history[-5:]):
                    if (
                        isinstance(event, CmdOutputObservation)
                        and hasattr(self, '_apply_patch_action_id')
                        and event.cause == self._apply_patch_action_id
                    ):
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
                return self._create_message_action(
                    content='Applying best attempt patch...',
                    wait_for_response=False,
                )

            prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
            best_attempt = self.attempt_storage.get_best_attempt(
                llm=self.llm, prompt_dir=prompt_dir
            )
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

        latest_user_message = state.get_last_user_message()
        if (
            latest_user_message
            and latest_user_message.content
            and latest_user_message.content.strip() == '/exit'
        ):
            return AgentFinishAction()

        if self.current_phase == Phase.ATTEMPT:
            for event in reversed(state.history[-10:]):
                if isinstance(event, BrowsePreviousAttemptsObservation):
                    for action_event in reversed(state.history):
                        if isinstance(action_event, BrowsePreviousAttemptsAction) and action_event.id == event.cause:
                            if self.browsing_state == BrowsingState.NONE:
                                self.browsing_state = BrowsingState.BROWSING
                                self.browsing_start_event_id = action_event.id
                                logger.info('Started browsing previous attempts during attempt')
                                self.tools = self._get_tools()
                                break
                            break
                    break

            if self.browsing_state == BrowsingState.BROWSING:
                for event in reversed(state.history[-5:]):
                    if isinstance(event, SubmitAttemptAsFinalAction):
                        attempt = self.attempt_storage.get_attempt(event.attempt_id)
                        if attempt:
                            logger.info(f'Submitting attempt {event.attempt_id} as final solution')
                            if self._pending_patch_application:
                                for obs_event in reversed(state.history[-5:]):
                                    if (
                                        isinstance(obs_event, CmdOutputObservation)
                                        and hasattr(self, '_apply_patch_action_id')
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
                        else:
                            logger.warning(f'Attempt {event.attempt_id} not found, cannot submit as final')
                            return self._create_message_action(
                                content=f'Attempt {event.attempt_id} not found. Cannot submit as final.',
                                wait_for_response=False,
                            )

                for event in reversed(state.history[-5:]):
                    if isinstance(event, FinishReflectionAction):
                        condensation_action = self._collapse_browsing_session(state)
                        self.browsing_state = BrowsingState.NONE
                        self.browsing_start_event_id = None
                        logger.info('Finished browsing previous attempts, resuming attempt')
                        self.tools = self._get_tools()

                        if condensation_action:
                            return condensation_action
                        else:
                            insights = event.message if event.message else 'No specific insights'
                            return self._create_message_action(
                                content=f'Reflection completed. Insights: {insights}. Resuming attempt.',
                                wait_for_response=False,
                            )

            for event in reversed(state.history[-10:]):
                if isinstance(event, FinishAttemptAction):
                    if self._pending_patch_extraction is not None:
                        if self._pending_patch_extraction[0] == event.id:
                            for obs_event in reversed(state.history[-5:]):
                                if (
                                    isinstance(obs_event, CmdOutputObservation)
                                    and obs_event.cause == self._pending_patch_extraction[1]
                                ):
                                    patch = obs_event.content
                                    if self.attempt_storage.current_attempt:
                                        self.attempt_storage.finish_attempt(
                                            end_event_id=event.id,
                                            summary=event.message,
                                            patch=patch,
                                        )
                                    self._pending_patch_extraction = None
                                    self._transition_to_next_phase()
                                    return self._create_message_action(
                                        content='Attempt completed. Transitioning to CHARACTERIZE phase to analyze and summarize this attempt.',
                                        wait_for_response=False,
                                    )
                            # Still waiting for patch extraction
                            return self._create_message_action(
                                content='Extracting patch for attempt...',
                                wait_for_response=False,
                            )

                    extract_cmd = None
                    if hasattr(self.config, 'extended') and self.config.extended:
                        extended_dict = self.config.extended.model_dump()
                        extract_cmd = extended_dict.get('rlm_extract_patch_cmd')
                    if extract_cmd and self.attempt_storage.current_attempt:
                        extract_action = CmdRunAction(command=extract_cmd)
                        extract_action.set_hard_timeout(600)
                        self.pending_actions.append(extract_action)
                        self._pending_patch_extraction = (event.id, extract_action.id)
                        logger.info(
                            f'Extracting patch for attempt {self.attempt_storage.current_attempt.attempt_id}'
                        )
                        return extract_action
                    else:
                        if self.attempt_storage.current_attempt:
                            self.attempt_storage.finish_attempt(
                                end_event_id=event.id, summary=event.message
                            )
                        self._transition_to_next_phase()
                        return self._create_message_action(
                            content='Attempt completed. Transitioning to CHARACTERIZE phase to analyze and summarize this attempt.',
                            wait_for_response=False,
                        )
        elif self.current_phase == Phase.CHARACTERIZE:
            for event in reversed(state.history[-10:]):
                if isinstance(event, FinishCharacterizationAction):
                    if self.attempt_storage.current_attempt:
                        self.attempt_storage.current_attempt.characterization_summary = (
                            event.characterization_summary
                        )
                    self._transition_to_next_phase()
                    return self._create_message_action(
                        content='Characterization completed. Transitioning to REFLECT phase to review all attempts and plan next steps.',
                        wait_for_response=False,
                    )
        else:  # Phase.REFLECT
            for event in reversed(state.history[-10:]):
                if isinstance(event, SubmitAttemptAsFinalAction):
                    attempt = self.attempt_storage.get_attempt(event.attempt_id)
                    if attempt:
                        logger.info(f'Submitting attempt {event.attempt_id} as final solution')
                        if self._pending_patch_application:
                            for obs_event in reversed(state.history[-5:]):
                                if (
                                    isinstance(obs_event, CmdOutputObservation)
                                    and hasattr(self, '_apply_patch_action_id')
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
                            logger.info(
                                f'Applying patch from submitted attempt {event.attempt_id}'
                            )
                            return apply_action
                        else:
                            return AgentFinishAction(
                                final_thought=f'Submitted attempt {event.attempt_id} as final solution. {event.message}'
                            )
                    else:
                        logger.warning(
                            f'Attempt {event.attempt_id} not found, cannot submit as final'
                        )
                        return self._create_message_action(
                            content=f'Attempt {event.attempt_id} not found. Cannot submit as final.',
                            wait_for_response=False,
                        )

            for event in reversed(state.history[-10:]):
                if isinstance(event, FinishReflectionAction):
                    self._transition_to_next_phase()
                    if state.history:
                        start_event_id = state.history[-1].id + 1
                    else:
                        start_event_id = 0
                    self.attempt_storage.start_attempt(
                        phase=Phase.ATTEMPT, start_event_id=start_event_id
                    )
                    return self._create_message_action(
                        content=f'Reflection phase completed. Starting new attempt phase (iteration {self.current_iteration}).',
                        wait_for_response=False,
                    )

        if self.current_phase == Phase.ATTEMPT and self.attempt_storage.current_attempt is None:
            if state.history:
                start_event_id = state.history[-1].id + 1
            else:
                start_event_id = 0
            self.attempt_storage.start_attempt(
                phase=Phase.ATTEMPT, start_event_id=start_event_id
            )

        for event in state.history:
            if isinstance(event, MessageAction):
                self._validate_action(event)

        if self.attempt_storage.current_attempt:
            for event in state.history[-10:]:
                if event.id >= self.attempt_storage.current_attempt.start_event_id:
                    if event not in self.attempt_storage.current_attempt.events:
                        self.attempt_storage.current_attempt.events.append(event)

        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        for event in condensed_history:
            if isinstance(event, MessageAction):
                self._validate_action(event)

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

        for action in actions:
            self._validate_action(action)

        for action in actions:
            if isinstance(action, BrowsePreviousAttemptsAction):
                summaries = self.attempt_storage.get_summarized_attempts()
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
            state: The current state (used for REFLECT phase to add attempt summaries)

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

        if self.current_phase == Phase.REFLECT:
            summaries = self.attempt_storage.get_summarized_attempts()
            if summaries:
                summary_text = 'Previous attempts summary:\n'
                for s in summaries:
                    summary_text += f"- {s['id']} ({s['phase']}): {s['characterization']}\n"

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def response_to_actions(self, response: 'ModelResponse') -> list['Action']:
        return rlm_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
        )

