from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_BROWSE_PREVIOUS_ATTEMPTS_TOOL_NAME = "browse_previous_attempts"
_BROWSE_PREVIOUS_ATTEMPTS_DESCRIPTION = """Browse information about previously submitted attempts.

Use this tool when:
- You need expanded details about previous attempts to fulfill the user's request.
"""

BrowsePreviousAttemptsTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=_BROWSE_PREVIOUS_ATTEMPTS_TOOL_NAME,
        description=_BROWSE_PREVIOUS_ATTEMPTS_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {},
            'required': [],
        },
    ),
)


