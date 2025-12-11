from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
import copy
from typing import TYPE_CHECKING

from openhands.llm.llm_registry import LLMRegistry

if TYPE_CHECKING:
    from litellm import ChatCompletionToolParam

    from openhands.events.action import Action
    from openhands.llm.llm import ModelResponse

import openhands.agenthub.rlm_agent.function_calling as rlm_function_calling
from openhands.agenthub.rlm_agent.tools import (
    BrowserTool,
    BrowsePreviousAttemptsTool,
    CondensationRequestTool,
    ExpandPreviousAttemptTool,
    FinishCharacterizationTool,
    FinishReflectionTool,
    FinishTool,
    IPythonTool,
    LLMBasedFileEditTool,
    SubmitAttemptAsFinalTool,
    ThinkTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)
from openhands.agenthub.rlm_agent.tools.task_tracker import create_task_tracker_tool
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import AgentFinishAction, CmdRunAction, MessageAction, SystemMessageAction
from openhands.events.action.agent import (
    BrowsePreviousAttemptsAction,
    ExpandPreviousAttemptAction,
    FinishAttemptAction,
    FinishCharacterizationAction,
    FinishReflectionAction,
    SubmitAttemptAsFinalAction,
)
from openhands.events.event import Event
from openhands.events.observation import CmdOutputObservation
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
    ATTEMPT = 'attempt'
    CHARACTERIZE = 'characterize'
    REFLECT = 'reflect'
    COMPLETE = 'complete'


@dataclass
class AttemptRecord:
    attempt_id: str
    iteration: int
    summary: str = ''
    characterization_title: str = ''
    characterization_summary: str = ''
    validation: str = ''
    confidence: str = ''
    limitations: str = ''
    reflection_plan: str = ''
    patch: str = ''
    patch_exit_code: int | None = None
    history_start_index: int | None = None
    history_end_index: int | None = None

    def summary_line(self) -> str:
        title = self.characterization_title or 'Untitled attempt'
        return f'{self.attempt_id}: {title} | summary: {self.summary or "n/a"}'


class RLMAgent(Agent):
    VERSION = '0.1'

    sandbox_plugins: list[PluginRequirement] = [
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry) -> None:
        super().__init__(config, llm_registry)
        self.pending_actions: deque['Action'] = deque()
        self.current_phase: Phase = Phase.ATTEMPT
        self.iteration_index: int = 1
        self.attempts: list[AttemptRecord] = []
        self._last_seen_event_id: int = -1
        self._pending_extract_attempt_id: str | None = None
        self._pending_extract_command: str | None = None
        self._pending_apply_attempt_id: str | None = None
        self._pending_apply_command: str | None = None

        self.rlm_max_iterations: int | None = self._resolve_max_iterations()
        self.extract_patch_cmd = self._resolve_optional_env(
            config, key='rlm_extract_patch_cmd', env_key='RLM_EXTRACT_PATCH_CMD'
        )
        self.apply_patch_cmd = self._resolve_optional_env(
            config, key='rlm_apply_patch_cmd', env_key='RLM_APPLY_PATCH_CMD'
        )

        self.phase_prompts = {
            Phase.ATTEMPT: 'system_prompt_attempt.j2',
            Phase.CHARACTERIZE: 'system_prompt_attempt.j2',
            Phase.REFLECT: 'system_prompt_reflect.j2',
        }
        self.characterize_transition_template = 'characterize_transition.j2'

        self.reset()
        self.tools = self._get_tools_for_phase(self.current_phase)

        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)
        self.condenser = Condenser.from_config(self.config.condenser, llm_registry)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        self.llm = self.llm_registry.get_router(self.config)

    @property
    def prompt_manager(self) -> PromptManager:
        if self._prompt_manager is None:
            self._prompt_manager = PromptManager(
                prompt_dir=os.path.join(os.path.dirname(__file__), 'prompts'),
                system_prompt_filename=self.phase_prompts[Phase.ATTEMPT],
            )
        return self._prompt_manager

    def _set_phase_prompt(self, phase: Phase) -> None:
        filename = self.phase_prompts.get(phase)
        if filename:
            self.prompt_manager.system_template = self.prompt_manager._load_template(
                filename
            )

    def _resolve_optional_env(
        self, config: AgentConfig, key: str, env_key: str
    ) -> str | None:
        value = None
        if config.extended and key in config.extended.model_dump():
            raw = config.extended.model_dump().get(key, '')
            value = str(raw) if raw else None
        if not value:
            env_val = os.environ.get(env_key, '')
            value = env_val if env_val else None
        return value

    def _resolve_max_iterations(self) -> int | None:
        if self.config.extended and 'rlm_max_iterations' in self.config.extended.model_dump():
            return int(self.config.extended['rlm_max_iterations'])
        env_val = os.environ.get('RLM_MAX_ITERATIONS')
        if env_val:
            return int(env_val)
        return None

    def reset(self) -> None:
        super().reset()
        self.pending_actions.clear()

    def _get_tools_for_phase(self, phase: Phase) -> list['ChatCompletionToolParam']:
        SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS = ['gpt-4', 'o3', 'o1', 'o4']

        use_short_tool_desc = False
        model_name = (
            getattr(getattr(self.llm, 'config', None), 'model', None)
            if self.llm is not None
            else None
        )
        if isinstance(model_name, str):
            use_short_tool_desc = any(
                model_substr in model_name
                for model_substr in SHORT_TOOL_DESCRIPTION_LLM_SUBSTRS
            )

        tools: list['ChatCompletionToolParam'] = []

        if phase == Phase.ATTEMPT:
            if self.config.enable_cmd:
                tools.append(
                    create_cmd_run_tool(use_short_description=use_short_tool_desc)
                )
            if self.config.enable_think:
                tools.append(ThinkTool)
            if self.config.enable_finish:
                tools.append(FinishTool)
            if self.config.enable_condensation_request:
                tools.append(CondensationRequestTool)
            if (
                self.config.enable_browsing
                and sys.platform != 'win32'
                and BrowserTool is not None
            ):
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
        elif phase == Phase.CHARACTERIZE:
            if self.config.enable_cmd:
                tools.append(
                    create_cmd_run_tool(use_short_description=use_short_tool_desc)
                )
            if self.config.enable_think:
                tools.append(ThinkTool)
            tools.append(FinishCharacterizationTool)
        elif phase == Phase.REFLECT:
            tools.append(BrowsePreviousAttemptsTool)
            tools.append(ExpandPreviousAttemptTool)
            if self.config.enable_think:
                tools.append(ThinkTool)
            tools.append(FinishReflectionTool)
            tools.append(SubmitAttemptAsFinalTool)

        return tools

    def _allowed_tool_names(self, tools: list['ChatCompletionToolParam']) -> set[str]:
        return {tool['function']['name'] for tool in tools}

    # Backwards-compat convenience for tests mirroring CodeActAgent
    def _get_tools(self) -> list['ChatCompletionToolParam']:
        return self._get_tools_for_phase(self.current_phase)

    def _phase_tools_and_names(self) -> tuple[list['ChatCompletionToolParam'], set[str]]:
        tools = self._get_tools_for_phase(self.current_phase)
        allowed = self._allowed_tool_names(tools)
        if self.config.enable_mcp and self.mcp_tools:
            allowed.update(self.mcp_tools.keys())
        return tools, allowed

    def _get_initial_user_message(self, history: list[Event]) -> MessageAction:
        initial_user_message: MessageAction | None = None
        for event in history:
            if isinstance(event, MessageAction) and event.source == 'user':
                initial_user_message = event
                break
        if initial_user_message is None:
            raise ValueError('Initial user message not found in history.')
        return initial_user_message

    def _get_messages(
        self, events: list[Event], initial_user_message: MessageAction
    ) -> list[Message]:
        messages = self.conversation_memory.process_events(
            condensed_history=events,
            initial_user_action=initial_user_message,
            max_message_chars=self.llm.config.max_message_chars,
            vision_is_active=self.llm.vision_is_active(),
        )

        if self.llm.is_caching_prompt_active():
            self.conversation_memory.apply_prompt_caching(messages)

        return messages

    def _build_reflect_messages(self) -> list[Message]:
        system_message = Message(
            role='system',
            content=[
                TextContent(
                    text=self.prompt_manager.get_system_message(
                        cli_mode=self.config.cli_mode
                    )
                )
            ],
            tool_calls=None,
        )
        summaries = '\n'.join([attempt.summary_line() for attempt in self.attempts]) or (
            'No attempts yet.'
        )
        summary_msg = Message(
            role='user', content=[TextContent(text=f'Attempt summaries so far:\n{summaries}')]
        )
        task_msg = Message(
            role='user',
            content=[
                TextContent(
                    text='Review attempts, expand as needed, then call finish_reflection(plan) or submit_attempt_as_final(attempt_id).'
                )
            ],
        )
        return [system_message, summary_msg, task_msg]

    def _process_new_observations(self, state: State) -> None:
        for event in state.history:
            if event.id is None or event.id <= self._last_seen_event_id:
                continue
            self._last_seen_event_id = event.id
            if isinstance(event, CmdOutputObservation):
                if (
                    self._pending_extract_attempt_id
                    and self._pending_extract_command
                    and event.command == self._pending_extract_command
                ):
                    attempt = self._find_attempt(self._pending_extract_attempt_id)
                    if attempt:
                        attempt.patch = event.content
                        attempt.patch_exit_code = event.exit_code
                    self._pending_extract_attempt_id = None
                    self._pending_extract_command = None
                if (
                    self._pending_apply_attempt_id
                    and self._pending_apply_command
                    and event.command == self._pending_apply_command
                ):
                    attempt = self._find_attempt(self._pending_apply_attempt_id)
                    if attempt:
                        attempt.patch_exit_code = event.exit_code
                    self._pending_apply_attempt_id = None
                    self._pending_apply_command = None

    def _find_attempt(self, attempt_id: str) -> AttemptRecord | None:
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        return None

    def _latest_attempt(self) -> AttemptRecord | None:
        return self.attempts[-1] if self.attempts else None

    def _guard_plain_reply(self, actions: list['Action']) -> list['Action']:
        if len(actions) == 1 and isinstance(actions[0], MessageAction):
            if self.current_phase == Phase.CHARACTERIZE:
                actions[0].content = (
                    'Please summarize this attempt with finish_characterization(title, summary, validation, confidence, limitations).'
                )
            elif self.current_phase == Phase.REFLECT:
                actions[0].content = (
                    'Use finish_reflection(plan) or submit_attempt_as_final(attempt_id) to conclude REFLECT.'
                )
        return actions

    def _queue_system_message(self, tools: list['ChatCompletionToolParam']) -> None:
        sys_msg = SystemMessageAction(
            content=self.prompt_manager.get_system_message(cli_mode=self.config.cli_mode),
            tools=tools,
            agent_class=self.__class__.__name__,
        )
        self.pending_actions.append(sys_msg)

    def _render_characterize_transition(
        self, attempt_id: str, attempt_summary: str
    ) -> str:
        try:
            template = self.prompt_manager._load_template(
                self.characterize_transition_template
            )
            return template.render(
                attempt_id=attempt_id, attempt_summary=attempt_summary
            ).strip()
        except FileNotFoundError:
            return (
                f'Characterize {attempt_id}. Use finish_characterization with title, summary, '
                f'validation, confidence, and limitations. Attempt summary: {attempt_summary}'
            )

    def _transition_to_phase(
        self, next_phase: Phase, transition_messages: list[str] | None = None
    ) -> None:
        self.current_phase = next_phase
        self._set_phase_prompt(next_phase)
        tools, _ = self._phase_tools_and_names()
        self.tools = tools
        if transition_messages:
            for msg in transition_messages:
                self.pending_actions.append(
                    MessageAction(content=msg, wait_for_response=True)
                )
        self._queue_system_message(tools)

    def _handle_finish_attempt(self, action: FinishAttemptAction) -> None:
        if not self.extract_patch_cmd:
            raise ValueError(
                'rlm_extract_patch_cmd is required to complete an attempt.'
            )
        attempt_id = f'attempt-{len(self.attempts) + 1}'
        record = AttemptRecord(
            attempt_id=attempt_id,
            iteration=self.iteration_index,
            summary=action.message,
            history_start_index=None,
        )
        self.attempts.append(record)

        self.pending_actions.append(action)

        self._pending_extract_attempt_id = attempt_id
        self._pending_extract_command = self.extract_patch_cmd
        self.pending_actions.append(
            MessageAction(
                content=f'Extracting patch for {attempt_id} via `{self.extract_patch_cmd}`.',
                wait_for_response=False,
            )
        )
        self.pending_actions.append(CmdRunAction(command=self.extract_patch_cmd))

        transition_msg = self._render_characterize_transition(
            attempt_id=attempt_id, attempt_summary=action.message
        )
        self._transition_to_phase(
            Phase.CHARACTERIZE, transition_messages=[transition_msg]
        )

    def _handle_finish_characterization(
        self, action: FinishCharacterizationAction
    ) -> None:
        attempt = self._latest_attempt()
        if attempt:
            attempt.characterization_title = action.characterization_title
            attempt.characterization_summary = action.characterization_summary
            if action.thought:
                for line in action.thought.splitlines():
                    if line.startswith('Validation: '):
                        attempt.validation = line.removeprefix('Validation: ').strip()
                    elif line.startswith('Confidence: '):
                        attempt.confidence = line.removeprefix('Confidence: ').strip()
                    elif line.startswith('Limitations: '):
                        attempt.limitations = line.removeprefix('Limitations: ').strip()

        self.pending_actions.append(action)

        summaries = [
            a.summary_line() for a in self.attempts if a.characterization_summary
        ]
        reflect_intro = (
            'Reflect on completed attempts. Use browse_previous_attempts to inspect details, '
            'then finish_reflection(plan) or submit_attempt_as_final(attempt_id).'
        )
        self._transition_to_phase(
            Phase.REFLECT,
            transition_messages=[
                'Completed attempt characterizations:\n' + '\n'.join(summaries),
                reflect_intro,
            ],
        )

    def _handle_finish_reflection(self, action: FinishReflectionAction) -> None:
        attempt = self._latest_attempt()
        if attempt:
            attempt.reflection_plan = action.final_message

        self.pending_actions.append(action)

        if self.rlm_max_iterations is not None and self.iteration_index >= self.rlm_max_iterations:
            best_attempt = self._select_best_attempt()
            self._apply_best_attempt_and_finish(best_attempt)
            return

        self.iteration_index += 1
        todo_msg = (
            f'Plan recorded. Starting ATTEMPT {self.iteration_index}/{self.rlm_max_iterations or "?"}. '
            'Proceed with tools and call finish when done.'
        )
        self._transition_to_phase(Phase.ATTEMPT, transition_messages=[todo_msg])

    def _handle_submit_attempt_as_final(
        self, action: SubmitAttemptAsFinalAction
    ) -> None:
        self.pending_actions.append(action)
        attempt = self._find_attempt(action.attempt_id)
        if attempt is None:
            self.pending_actions.append(
                MessageAction(
                    content=f'Unknown attempt id {action.attempt_id}.',
                    wait_for_response=True,
                )
            )
            return
        self._apply_best_attempt_and_finish(attempt)

    def _handle_browse_attempts(self, action: BrowsePreviousAttemptsAction) -> None:
        filters = (
            [item for item in action.thought.split(',') if item] if action.thought else []
        )
        attempts = (
            [a for a in self.attempts if a.attempt_id in filters]
            if filters
            else self.attempts
        )
        lines = []
        for attempt in attempts:
            line = attempt.summary_line()
            if attempt.characterization_summary:
                line += f'\n characterization: {attempt.characterization_summary}'
            if attempt.validation:
                line += f'\n validation: {attempt.validation}'
            lines.append(line)
        action.content = '\n\n'.join(lines) if lines else 'No attempts available.'
        self.pending_actions.append(action)

    def _handle_expand_attempt(self, action: ExpandPreviousAttemptAction) -> None:
        attempt = self._find_attempt(action.attempt_id)
        if attempt is None:
            action.content = f'Attempt {action.attempt_id} not found.'
        else:
            detail = [
                attempt.summary_line(),
                f'Characterization: {attempt.characterization_summary}',
                f'Validation: {attempt.validation or "n/a"}',
                f'Confidence: {attempt.confidence or "n/a"}',
                f'Limitations: {attempt.limitations or "n/a"}',
            ]
            if attempt.patch:
                patch_preview = attempt.patch
                if len(patch_preview) > 2000:
                    patch_preview = (
                        patch_preview[:1000]
                        + '\n...[patch truncated]...\n'
                        + patch_preview[-1000:]
                    )
                detail.append(f'Patch:\n{patch_preview}')
            action.content = '\n'.join(detail)
        self.pending_actions.append(action)

    def _apply_best_attempt_and_finish(self, attempt: AttemptRecord) -> None:
        if not attempt.patch:
            raise ValueError(
                f'No extracted patch found for {attempt.attempt_id}. Cannot apply.'
            )
        if not self.apply_patch_cmd:
            raise ValueError(
                'rlm_apply_patch_cmd is required to apply the best attempt.'
            )
        apply_cmd = self._build_apply_command(attempt.patch)
        self._pending_apply_attempt_id = attempt.attempt_id
        self._pending_apply_command = apply_cmd
        self.pending_actions.append(
            MessageAction(
                content=f'Applying patch for {attempt.attempt_id} via `{self.apply_patch_cmd}`.',
                wait_for_response=False,
            )
        )
        self.pending_actions.append(CmdRunAction(command=apply_cmd))

        final_summary = (
            f'Applied attempt {attempt.attempt_id}. '
            f'Title: {attempt.characterization_title or "n/a"}. '
            f'Summary: {attempt.characterization_summary or attempt.summary}'
        )
        # TODO: persist attempt metadata to disk for browsing across sessions.
        self.current_phase = Phase.COMPLETE
        self.pending_actions.append(AgentFinishAction(final_thought=final_summary))

    def _build_apply_command(self, patch: str) -> str:
        return f"cat <<'PATCH' | {self.apply_patch_cmd}\n{patch}\nPATCH"

    def _select_best_attempt(self) -> AttemptRecord:
        completed = [
            a for a in self.attempts if a.characterization_summary or a.summary
        ]
        if not completed:
            raise ValueError('No completed attempts to select from.')
        if len(completed) == 1:
            return completed[0]

        try:
            summaries = '\n'.join(
                f'{a.attempt_id}: {a.characterization_title or a.summary} | validation={a.validation or "n/a"} | confidence={a.confidence or "n/a"}'
                for a in completed
            )
            messages = [
                {'role': 'system', 'content': 'Select the best attempt id given summaries.'},
                {'role': 'user', 'content': summaries},
            ]
            resp = self.llm.completion(
                messages=messages,
                temperature=0,
                log_metadata={'phase': self.current_phase.value},
            )
            content = (
                resp.choices[0].message.content
                if resp and resp.choices and resp.choices[0].message
                else ''
            )
            for attempt in completed:
                if attempt.attempt_id in str(content):
                    return attempt
        except Exception as e:
            logger.warning(f'LLM selection failed, using fallback: {e}')
        return completed[-1]

    def step(self, state: State) -> 'Action':
        if self.pending_actions:
            action = self.pending_actions.popleft()
            if action.id != Event.INVALID_ID:
                logger.debug(
                    'Resetting pending action id before emit: %s (type=%s, phase=%s)',
                    action.id,
                    type(action).__name__,
                    self.current_phase,
                )
                action.id = Event.INVALID_ID
            return action

        self._process_new_observations(state)

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        if self.rlm_max_iterations is None and hasattr(state, 'iteration_flag'):
            runtime_max = getattr(state.iteration_flag, 'max_value', None)
            # Only honor runtime/CLI if explicitly set; otherwise default to 3
            if runtime_max and runtime_max != 100:
                self.rlm_max_iterations = runtime_max
            else:
                self.rlm_max_iterations = 3
        if self.rlm_max_iterations is None:
            self.rlm_max_iterations = 3

        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events
            case Condensation(action=condensation_action):
                cloned = copy.deepcopy(condensation_action)
                logger.debug(
                    'Cloning condensation action to clear id: %s (type=%s, phase=%s)',
                    condensation_action.id,
                    type(condensation_action).__name__,
                    self.current_phase,
                )
                cloned.id = Event.INVALID_ID
                return cloned

        initial_user_message = self._get_initial_user_message(state.history)

        if self.current_phase == Phase.CHARACTERIZE and not self._latest_attempt():
            self._transition_to_phase(
                Phase.ATTEMPT,
                transition_messages=[
                    'No finished attempt to characterize. Return to ATTEMPT and call finish when ready.'
                ],
            )
            action = self.pending_actions.popleft()
            if action.id != Event.INVALID_ID:
                logger.debug(
                    'Resetting bounce-back action id before emit: %s (type=%s, phase=%s)',
                    action.id,
                    type(action).__name__,
                    self.current_phase,
                )
                action.id = Event.INVALID_ID
            return action

        if (
            self.current_phase == Phase.REFLECT
            and len([a for a in self.attempts if a.summary]) == 0
        ):
            self._transition_to_phase(
                Phase.ATTEMPT,
                transition_messages=['No attempts yet. Start ATTEMPT and call finish.'],
            )
            action = self.pending_actions.popleft()
            if action.id != Event.INVALID_ID:
                logger.debug(
                    'Resetting reflect-bounce action id before emit: %s (type=%s, phase=%s)',
                    action.id,
                    type(action).__name__,
                    self.current_phase,
                )
                action.id = Event.INVALID_ID
            return action

        self._set_phase_prompt(self.current_phase)

        if self.current_phase == Phase.REFLECT:
            messages = self._build_reflect_messages()
        else:
            messages = self._get_messages(condensed_history, initial_user_message)

        tools, allowed_tool_names = self._phase_tools_and_names()
        params: dict = {
            'messages': messages,
            'tools': check_tools(tools, self.llm.config),
            'extra_body': {
                'metadata': state.to_llm_metadata(
                    model_name=self.llm.config.model, agent_name=self.name
                )
            },
            'log_metadata': {'phase': self.current_phase.value},
        }
        response = self.llm.completion(**params)
        logger.debug(f'Response from LLM: {response}')
        actions = rlm_function_calling.response_to_actions(
            response,
            mcp_tool_names=list(self.mcp_tools.keys()),
            allowed_tool_names=allowed_tool_names,
        )
        actions = self._guard_plain_reply(actions)
        logger.debug(f'Actions after response_to_actions: {actions}')

        for action in actions:
            if isinstance(action, FinishAttemptAction):
                self._handle_finish_attempt(action)
            elif isinstance(action, FinishCharacterizationAction):
                self._handle_finish_characterization(action)
            elif isinstance(action, FinishReflectionAction):
                self._handle_finish_reflection(action)
            elif isinstance(action, SubmitAttemptAsFinalAction):
                self._handle_submit_attempt_as_final(action)
            elif isinstance(action, BrowsePreviousAttemptsAction):
                self._handle_browse_attempts(action)
            elif isinstance(action, ExpandPreviousAttemptAction):
                self._handle_expand_attempt(action)
            else:
                self.pending_actions.append(action)

        action = self.pending_actions.popleft()
        if action.id != Event.INVALID_ID:
            logger.debug(
                'Resetting final pending action id before emit: %s (type=%s, phase=%s)',
                action.id,
                type(action).__name__,
                self.current_phase,
            )
            action.id = Event.INVALID_ID
        return action


