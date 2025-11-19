from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_EXPAND_PREVIOUS_ATTEMPT_TOOL_NAME = "expand_previous_attempt"
_EXPAND_PREVIOUS_ATTEMPT_DESCRIPTION = """Expand on a previous attempt with more details.

Use this tool when:
- You need to provide additional information or context about a previous attempt.
- You want to refine or build upon a previous solution.
"""

ExpandPreviousAttemptTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=_EXPAND_PREVIOUS_ATTEMPT_TOOL_NAME,
        description=_EXPAND_PREVIOUS_ATTEMPT_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['id'],
            'properties': {
                'id': {
                    'type': 'string',
                    'description': 'The unique identifier of the previous attempt to expand upon',
                },
            },
        },
    ),
)
