from unittest.mock import Mock, patch

import pytest

from litellm import ModelResponse

from openhands.agenthub.rlm_agent.rlm_agent import Phase, RLMAgent
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.exceptions import FunctionCallNotExistsError
from openhands.core.message import Message, TextContent
from openhands.events.action import MessageAction
from openhands.events.event import EventSource
from openhands.memory.condenser import View


class DummyState:
    """Minimal state stub for RLMAgent.step tests."""

    def __init__(self, history):
        self.history = history
        self.extra_data = {}

    def get_last_user_message(self):
        for event in reversed(self.history):
            if isinstance(event, MessageAction):
                source = getattr(event, 'source', getattr(event, '_source', None))
                if source == EventSource.USER:
                    return event
        return None

    def to_llm_metadata(self, **_: str):
        return {}


def _make_plain_response() -> ModelResponse:
    """LLM response with no tool calls and empty content."""
    return ModelResponse(
        id='resp',
        choices=[
            {
                'message': {'role': 'assistant', 'content': '', 'tool_calls': None},
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )


def _build_agent(monkeypatch, model: str = 'gpt-4o') -> RLMAgent:
    """Create an RLMAgent with patched dependencies and a mocked LLM."""
    llm_config = LLMConfig(model=model, api_key='test_key')
    config = AgentConfig(
        enable_cmd=False,
        enable_editor=False,
        enable_llm_editor=False,
        enable_think=False,
        enable_jupyter=False,
        enable_browsing=False,
        enable_plan_mode=False,
        enable_condensation_request=False,
    )

    mock_llm = Mock()
    mock_llm.config = llm_config
    mock_llm.is_caching_prompt_active.return_value = False
    mock_llm.vision_is_active.return_value = False
    mock_llm.completion.return_value = _make_plain_response()

    registry = Mock()
    registry.get_router = Mock(return_value=mock_llm)

    with patch('openhands.agenthub.rlm_agent.rlm_agent.Condenser.from_config') as mock_cond:
        mock_condenser = Mock()
        mock_condenser.condensed_history.return_value = View(events=[])
        mock_cond.return_value = mock_condenser
        agent = RLMAgent(config=config, llm_registry=registry)

    # Simplify conversation memory to avoid prompt processing details.
    agent.conversation_memory = Mock()
    agent.conversation_memory.process_events.return_value = [
        Message(role='user', content=[TextContent(text='task')])
    ]
    agent.conversation_memory.apply_prompt_caching = lambda *_args, **_kwargs: None
    return agent


def _build_state() -> DummyState:
    user_msg = MessageAction(content='Initial user message')
    user_msg._source = EventSource.USER
    user_msg.id = 0
    return DummyState(history=[user_msg])


def test_reflect_plain_reply_is_nudged(monkeypatch):
    """Plain text in REFLECT should be replaced with a tool-call nudge."""
    agent = _build_agent(monkeypatch)
    agent.current_phase = Phase.REFLECT
    agent.tools = agent._get_tools()
    state = _build_state()

    action = agent.step(state)

    assert isinstance(action, MessageAction)
    assert 'finish_reflection' in action.content.lower() or 'submit_attempt_as_final' in action.content


def test_characterize_plain_reply_is_nudged(monkeypatch):
    """Plain text in CHARACTERIZE should be replaced with a finish_characterization nudge."""
    agent = _build_agent(monkeypatch)
    agent.current_phase = Phase.CHARACTERIZE
    agent.tools = agent._get_tools()
    state = _build_state()

    action = agent.step(state)

    assert isinstance(action, MessageAction)
    assert 'finish_characterization' in action.content


def test_characterize_without_attempt_bounces_back(monkeypatch):
    """If no finished attempt exists, CHARACTERIZE should be blocked and reverted."""
    agent = _build_agent(monkeypatch)
    agent.current_phase = Phase.CHARACTERIZE
    agent.tools = agent._get_tools()
    state = _build_state()

    action = agent.step(state)

    assert agent.current_phase == Phase.ATTEMPT
    assert isinstance(action, MessageAction)
    assert 'finish_attempt' in action.content


def test_characterize_rejects_disallowed_tool(monkeypatch):
    """Characterize phase should reject tools that are not offered."""
    agent = _build_agent(monkeypatch)
    agent.current_phase = Phase.CHARACTERIZE
    agent.tools = agent._get_tools()
    state = _build_state()

    # Simulate an LLM response that tries to call an editing tool that is not
    # part of the CHARACTERIZE toolset.
    agent.llm.completion.return_value = ModelResponse(
        id='resp',
        choices=[
            {
                'message': {
                    'role': 'assistant',
                    'content': '',
                    'tool_calls': [
                        {
                            'id': 'call_1',
                            'type': 'function',
                            'function': {
                                'name': 'str_replace_editor',
                                'arguments': '{}',
                            },
                        }
                    ],
                },
                'finish_reason': 'tool_calls',
                'index': 0,
            }
        ],
    )

    with pytest.raises(FunctionCallNotExistsError) as exc_info:
        agent.step(state)

    assert 'str_replace_editor' in str(exc_info.value)
    assert 'Allowed tools' in str(exc_info.value)

