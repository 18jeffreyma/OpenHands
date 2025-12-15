from __future__ import annotations

import os
import re
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
        self._expanded_attempt_ids: set[str] = set()  # Track which attempts have been expanded to prevent loops

        self.rlm_max_iterations: int | None = self._resolve_max_iterations()
        self.extract_patch_cmd = self._resolve_optional_env(
            config, key='rlm_extract_patch_cmd', env_key='RLM_EXTRACT_PATCH_CMD'
        )
        self.apply_patch_cmd = self._resolve_optional_env(
            config, key='rlm_apply_patch_cmd', env_key='RLM_APPLY_PATCH_CMD'
        )
        self.reset_repo_cmd = self._resolve_optional_env(
            config, key='rlm_reset_cmd', env_key='RLM_RESET_CMD'
        )

        # Reminder mechanism: track steps in ATTEMPT phase and remind periodically
        self._attempt_phase_step_count: int = 0
        self._reminder_step_interval: int = 5  # Default: remind every 5 steps (0 to disable) - more frequent to catch optimization loops

        attempt_prompt = self.config.resolved_system_prompt_filename
        self.phase_prompts = {
            Phase.ATTEMPT: attempt_prompt,
            Phase.CHARACTERIZE: attempt_prompt,
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
        # default to 1 iteration if not specified
        return 1

    def reset(self) -> None:
        super().reset()
        self.pending_actions.clear()
        self._attempt_phase_step_count = 0
        self._expanded_attempt_ids.clear()

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
        # Filter out patch extraction/application commands and their observations from LLM messages
        # Also filter reset commands to avoid tool metadata issues
        filtered_events = []
        for event in events:
            should_filter = False
            if isinstance(event, (CmdRunAction, CmdOutputObservation)):
                event_cmd = event.command

                # Check extract command: match against pending (expanded), original template, or expanded template
                if self.extract_patch_cmd:
                    if (self._pending_extract_command and event_cmd == self._pending_extract_command) or \
                       event_cmd == self.extract_patch_cmd or \
                       event_cmd == self._expand_shell_vars(self.extract_patch_cmd):
                        should_filter = True
                    # Fallback: match by base pattern "git diff"
                    if not should_filter:
                        extract_base = self.extract_patch_cmd.split()[0:2] if self.extract_patch_cmd else []
                        event_base = event_cmd.split()[0:2] if event_cmd else []
                        if extract_base and event_base and len(extract_base) == 2 and extract_base == event_base:
                            if extract_base[0] == 'git' and extract_base[1] == 'diff':
                                should_filter = True

                # Check apply command: match against pending (expanded), original template, or expanded template
                if not should_filter and self.apply_patch_cmd:
                    if (self._pending_apply_command and event_cmd == self._pending_apply_command) or \
                       event_cmd == self.apply_patch_cmd or \
                       event_cmd == self._expand_shell_vars(self.apply_patch_cmd):
                        should_filter = True
                    # Fallback: match by base pattern "git apply"
                    if not should_filter:
                        apply_base = self.apply_patch_cmd.split()[0:2] if self.apply_patch_cmd else []
                        event_base = event_cmd.split()[0:2] if event_cmd else []
                        if apply_base and event_base and len(apply_base) == 2 and apply_base == event_base:
                            if apply_base[0] == 'git' and apply_base[1] == 'apply':
                                should_filter = True

                # Check reset command: match against original template or expanded template
                if not should_filter and self.reset_repo_cmd:
                    reset_expanded = self._expand_shell_vars(self.reset_repo_cmd)
                    if event_cmd == self.reset_repo_cmd or event_cmd == reset_expanded:
                        should_filter = True
                    # Fallback: match by base pattern "git reset"
                    if not should_filter:
                        reset_base = reset_expanded.split()[0:2] if reset_expanded else []
                        event_base = event_cmd.split()[0:2] if event_cmd else []
                        if reset_base and event_base and len(reset_base) == 2 and reset_base == event_base:
                            if reset_base[0] == 'git' and reset_base[1] == 'reset':
                                should_filter = True

            if not should_filter:
                filtered_events.append(event)

        messages = self.conversation_memory.process_events(
            condensed_history=filtered_events,
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
                        cli_mode=self.config.cli_mode,
                        rlm_attempt_phase=False,  # REFLECT phase, not ATTEMPT
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
        # Filter out empty text content from messages
        messages = [system_message, summary_msg, task_msg]
        filtered_messages = []
        for msg in messages:
            filtered_content = [
                item
                for item in msg.content
                if not (isinstance(item, TextContent) and item.text == '')
            ]
            if filtered_content:
                filtered_msg = Message(
                    role=msg.role,
                    content=filtered_content,
                    cache_enabled=msg.cache_enabled,
                    vision_enabled=msg.vision_enabled,
                    function_calling_enabled=msg.function_calling_enabled,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    force_string_serializer=msg.force_string_serializer,
                )
                filtered_messages.append(filtered_msg)
        return filtered_messages

    def _process_new_observations(self, state: State) -> None:
        for event in state.history:
            if event.id is None or event.id <= self._last_seen_event_id:
                continue
            self._last_seen_event_id = event.id
            if isinstance(event, CmdOutputObservation):
                if (
                    self._pending_extract_attempt_id
                    and self._pending_extract_command
                ):
                    # Match commands - handle environment variable expansion differences
                    # Match if exact match, or if both start with same base command (first 2 words)
                    pending_parts = self._pending_extract_command.split()[:2] if self._pending_extract_command else []
                    obs_parts = event.command.split()[:2] if event.command else []
                    commands_match = (
                        event.command == self._pending_extract_command
                        or (pending_parts and obs_parts and pending_parts == obs_parts)
                    )

                    if commands_match:
                        attempt = self._find_attempt(self._pending_extract_attempt_id)
                        if attempt:
                            attempt.patch = event.content
                            attempt.patch_exit_code = event.exit_code
                            logger.info(
                                'Processed patch extraction for %s (exit_code=%s, patch_length=%d)',
                                self._pending_extract_attempt_id,
                                event.exit_code,
                                len(event.content) if event.content else 0,
                            )
                        else:
                            logger.warning(
                                'Patch extraction observation received but attempt %s not found',
                                self._pending_extract_attempt_id,
                            )
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

                # Log git reset commands for debugging
                if event.command and 'git reset' in event.command:
                    logger.info(
                        'Git reset command observation: command=%s, exit_code=%s, content_length=%d',
                        event.command,
                        event.exit_code,
                        len(event.content) if event.content else 0,
                    )

    def _find_attempt(self, attempt_id: str) -> AttemptRecord | None:
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        return None

    def _latest_attempt(self) -> AttemptRecord | None:
        return self.attempts[-1] if self.attempts else None

    def _queue_system_message(self, tools: list['ChatCompletionToolParam']) -> None:
        sys_msg = SystemMessageAction(
            content=self.prompt_manager.get_system_message(
                cli_mode=self.config.cli_mode,
                rlm_attempt_phase=(self.current_phase == Phase.ATTEMPT),
            ),
            tools=tools,
            agent_class=self.__class__.__name__,
        )
        self.pending_actions.append(sys_msg)

    def _queue_repo_reset(self) -> None:
        if not self.reset_repo_cmd:
            logger.debug('No reset_repo_cmd configured, skipping repository reset')
            return
        # Expand shell variables in Python to avoid expansion issues
        original_cmd = self.reset_repo_cmd
        reset_cmd = self._expand_shell_vars(self.reset_repo_cmd)
        logger.info(
            'Resetting repository before ATTEMPT: original=`%s`, expanded=`%s`, cmd_length=%d',
            original_cmd,
            reset_cmd,
            len(reset_cmd),
        )
        # Use blocking=False to match evaluation script pattern
        # This uses NO_CHANGE_TIMEOUT mechanism instead of waiting indefinitely for PS1
        reset_action = CmdRunAction(command=reset_cmd, hidden=True)
        # Use 120 second timeout (git reset can be slow on large repos) with blocking=False
        # to allow NO_CHANGE_TIMEOUT to trigger if the command produces no output
        reset_action.set_hard_timeout(120, blocking=False)
        self.pending_actions.append(reset_action)

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
        self, next_phase: Phase, transition_messages: list[str] | None = None, state: State | None = None
    ) -> None:
        old_phase = self.current_phase
        self.current_phase = next_phase
        # Update state.extra_data immediately if state is provided
        # This ensures external code (e.g., evaluation scripts) can read the current phase
        if state is not None:
            state.extra_data['rlm_phase'] = self.current_phase.value

        # Reset reminder counter when transitioning away from ATTEMPT phase
        if old_phase == Phase.ATTEMPT and next_phase != Phase.ATTEMPT:
            self._attempt_phase_step_count = 0
        # Reset counter when entering ATTEMPT phase
        elif old_phase != Phase.ATTEMPT and next_phase == Phase.ATTEMPT:
            self._attempt_phase_step_count = 0

        # Check if the prompt template changes between phases
        old_prompt = self.phase_prompts.get(old_phase)
        new_prompt = self.phase_prompts.get(next_phase)
        prompt_changed = old_prompt != new_prompt

        self._set_phase_prompt(next_phase)
        tools, _ = self._phase_tools_and_names()
        self.tools = tools

        if transition_messages:
            for msg in transition_messages:
                self.pending_actions.append(
                    MessageAction(content=msg, wait_for_response=False)
                )

        # Only create a new SystemMessageAction if the prompt template actually changed
        # and we're NOT transitioning to REFLECT phase (which uses _build_reflect_messages)
        # For CHARACTERIZE phase (which uses the same prompt as ATTEMPT), we keep the
        # existing system message and just update the tools, preserving conversation context
        # For REFLECT phase, _build_reflect_messages() creates its own system message
        if prompt_changed and next_phase != Phase.REFLECT:
            self._queue_system_message(tools)

    def _handle_finish_attempt(self, action: FinishAttemptAction, state: State | None = None) -> None:
        attempt_id = f'attempt-{len(self.attempts) + 1}'
        record = AttemptRecord(
            attempt_id=attempt_id,
            iteration=self.iteration_index,
            summary=action.message,
            history_start_index=None,
        )
        self.attempts.append(record)

        # Reset reminder counter when finish is called
        self._attempt_phase_step_count = 0

        self.pending_actions.append(action)

        if self.extract_patch_cmd:
            # Run in the runtime via CmdRunAction (filtered from LLM messages but visible to agent processing)
            # Expand shell variables in Python to avoid expansion issues
            extract_cmd = self._expand_shell_vars(self.extract_patch_cmd)
            self._pending_extract_attempt_id = attempt_id
            self._pending_extract_command = extract_cmd
            # Use blocking=False to match evaluation script pattern
            # This uses NO_CHANGE_TIMEOUT mechanism instead of waiting indefinitely for PS1
            extract_action = CmdRunAction(command=extract_cmd, hidden=False)
            # Use 120 second timeout (git diff can be large) with blocking=False
            extract_action.set_hard_timeout(120, blocking=False)
            self.pending_actions.append(extract_action)
            logger.info(
                'Extracting patch for %s via `%s`.',
                attempt_id,
                extract_cmd,
            )
        else:
            logger.info(
                'No rlm_extract_patch_cmd configured; skipping patch extraction for %s.',
                attempt_id,
            )

        transition_msg = self._render_characterize_transition(
            attempt_id=attempt_id, attempt_summary=action.message
        )
        self._transition_to_phase(
            Phase.CHARACTERIZE, transition_messages=[transition_msg], state=state
        )

    def _handle_finish_characterization(
        self, action: FinishCharacterizationAction, state: State | None = None
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
            state=state,
        )

    def _handle_finish_reflection(self, action: FinishReflectionAction, state: State | None = None) -> None:
        attempt = self._latest_attempt()
        if attempt:
            attempt.reflection_plan = action.final_message

        self.pending_actions.append(action)

        if self.rlm_max_iterations is not None and self.iteration_index >= self.rlm_max_iterations:
            best_attempt = self._select_best_attempt()
            self._apply_best_attempt_and_finish(best_attempt)
            return

        self.iteration_index += 1
        self._queue_repo_reset()
        todo_msg = (
            f'Plan recorded. Starting ATTEMPT {self.iteration_index}/{self.rlm_max_iterations or "?"}. '
            'Proceed with tools and call finish when done.'
        )
        self._transition_to_phase(Phase.ATTEMPT, transition_messages=[todo_msg], state=state)

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
        content = '\n\n'.join(lines) if lines else 'No attempts available.'

        # If in REFLECT phase and all attempts have been expanded, add a strong reminder
        if self.current_phase == Phase.REFLECT and len(self._expanded_attempt_ids) >= len(self.attempts):
            content += (
                '\n\n⚠️ REMINDER: You have already reviewed all attempts. '
                'You MUST call `finish_reflection(plan)` NOW with your plan for the next attempt, '
                'or `submit_attempt_as_final(attempt_id)` if an attempt succeeded. '
                'Do NOT call browse_previous_attempts or expand_previous_attempt again.'
            )

        action.content = content
        self.pending_actions.append(action)

    def _handle_expand_attempt(self, action: ExpandPreviousAttemptAction) -> None:
        attempt = self._find_attempt(action.attempt_id)
        if attempt is None:
            action.content = f'Attempt {action.attempt_id} not found.'
        elif action.attempt_id in self._expanded_attempt_ids:
            # Prevent repeated expansion of the same attempt to avoid loops
            action.content = (
                f'Attempt {action.attempt_id} has already been expanded. '
                f'You have already seen the full details. Please call `finish_reflection` with a plan '
                f'or `submit_attempt_as_final` if this attempt succeeded.'
            )
        else:
            self._expanded_attempt_ids.add(action.attempt_id)
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

    def _expand_shell_vars(self, cmd: str) -> str:
        """Expand shell variable syntax like ${VAR:-default} in command string."""
        # Pattern to match ${VAR:-default} syntax
        pattern = r'\$\{([^:}]+):-([^}]+)\}'

        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2)
            # Get value from environment or use default
            env_value = os.environ.get(var_name)
            expanded_value = env_value if env_value else default_value
            return expanded_value

        expanded_cmd = re.sub(pattern, replace_var, cmd)

        # Remove quotes around simple expanded values (like commit hashes)
        # Pattern: "value" where value is alphanumeric/hex (commit hash)
        # This handles cases like: git reset --hard "${VAR:-hash}" -> git reset --hard hash
        # Match quoted strings that are alphanumeric/hex (typical git commit hashes)
        # Minimum 7 chars (short commit hash) up to 40 chars (full commit hash)
        quote_pattern = r'"([a-f0-9]{7,40})"'  # Match quoted hex strings (git commit hashes)
        expanded_cmd = re.sub(quote_pattern, r'\1', expanded_cmd)

        return expanded_cmd

    def _apply_best_attempt_and_finish(self, attempt: AttemptRecord) -> None:
        if not attempt.patch:
            raise ValueError(
                f'No extracted patch found for {attempt.attempt_id}. Cannot apply.'
            )
        if not self.apply_patch_cmd:
            raise ValueError(
                'rlm_apply_patch_cmd is required to apply the best attempt.'
            )
        # Reset repository to clean state before applying patch
        # Use git reset --hard only (no git clean) for faster, sufficient reset
        # git clean -fd can be slow and isn't necessary for patch application
        if self.reset_repo_cmd:
            # Extract just the git reset part, avoiding git clean
            reset_cmd = self.reset_repo_cmd.split('&&')[0].strip()
            if 'git reset --hard' in reset_cmd:
                # Expand shell variables in Python to avoid expansion issues
                original_reset_cmd = reset_cmd
                reset_cmd = self._expand_shell_vars(reset_cmd)
                logger.info(
                    'Resetting repository before applying patch: original=`%s`, expanded=`%s`, cmd_length=%d',
                    original_reset_cmd,
                    reset_cmd,
                    len(reset_cmd),
                )
                # Use blocking=False to match evaluation script pattern
                # This uses NO_CHANGE_TIMEOUT mechanism instead of waiting indefinitely for PS1
                reset_action = CmdRunAction(command=reset_cmd, hidden=True)
                # Use 120 second timeout (git reset can be slow on large repos) with blocking=False
                # to allow NO_CHANGE_TIMEOUT to trigger if the command produces no output
                reset_action.set_hard_timeout(120, blocking=False)
                self.pending_actions.append(reset_action)
        apply_cmd = self._build_apply_command(attempt.patch)
        self._pending_apply_attempt_id = attempt.attempt_id
        self._pending_apply_command = apply_cmd
        self.pending_actions.append(
            MessageAction(
                content=f'Applying patch for {attempt.attempt_id} via `{self.apply_patch_cmd}`.',
                wait_for_response=False,
            )
        )
        # Use blocking=False to match evaluation script pattern
        # This uses NO_CHANGE_TIMEOUT mechanism instead of waiting indefinitely for PS1
        apply_action = CmdRunAction(command=apply_cmd, hidden=False)
        # Use 120 second timeout (git apply for large patches) with blocking=False
        apply_action.set_hard_timeout(120, blocking=False)
        self.pending_actions.append(apply_action)

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
        # Sync current phase to state.extra_data for external phase detection
        # This allows evaluation scripts and other code to reliably detect the current phase
        state.extra_data['rlm_phase'] = self.current_phase.value

        # Always process new observations first, even if there are pending actions
        # This ensures patch extraction observations are processed immediately
        self._process_new_observations(state)

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

        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        if self.rlm_max_iterations is None and hasattr(state, 'iteration_flag'):
            runtime_max = getattr(state.iteration_flag, 'max_value', None)
            # Honor runtime/CLI if explicitly set; otherwise default to 1
            if runtime_max and runtime_max != 100:
                self.rlm_max_iterations = runtime_max
            else:
                self.rlm_max_iterations = 1
        if self.rlm_max_iterations is None:
            self.rlm_max_iterations = 1

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
                state=state,
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
                state=state,
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

        # Track steps in ATTEMPT phase and inject reminder if needed
        if self.current_phase == Phase.ATTEMPT:
            self._attempt_phase_step_count += 1

            # Inject reminder if interval reached and reminders are enabled
            if (
                self._reminder_step_interval > 0
                and self._attempt_phase_step_count % self._reminder_step_interval == 0
            ):
                reminder_msg = (
                    f"⚠️ REMINDER: You've taken {self._attempt_phase_step_count} steps in ATTEMPT phase. "
                    f"If you've run ANY test/workload and seen results (even if they show improvement), "
                    f"you MUST call `finish` NOW. Do NOT continue optimizing - ONE attempt = ONE test run + finish. "
                    f"The REFLECT phase will handle planning further improvements."
                )
                # Inject reminder as a user message in the messages list
                reminder_message = Message(
                    role='user',
                    content=[TextContent(text=reminder_msg)],
                )
                messages.append(reminder_message)

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
        logger.debug(f'Actions after response_to_actions: {actions}')

        for action in actions:
            if isinstance(action, FinishAttemptAction):
                self._handle_finish_attempt(action, state=state)
            elif isinstance(action, FinishCharacterizationAction):
                self._handle_finish_characterization(action, state=state)
            elif isinstance(action, FinishReflectionAction):
                self._handle_finish_reflection(action, state=state)
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


