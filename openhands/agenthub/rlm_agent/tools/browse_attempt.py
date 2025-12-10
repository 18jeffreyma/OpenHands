from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_BROWSE_ATTEMPT_TOOL_NAME = "browse_attempt"
_BROWSE_ATTEMPT_DESCRIPTION = """Browse the full trajectory of a specific attempt.

Use this tool when:
- You want to see the complete sequence of actions and events from a specific attempt.
- You need detailed information about what was done in a previous attempt beyond the summary.
- You want to understand the step-by-step process of a previous attempt.

The summaries of all attempts are already provided in the reflection phase. Use this tool to expand a specific attempt ID to see its full trajectory.
"""

BrowseAttemptTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=_BROWSE_ATTEMPT_TOOL_NAME,
        description=_BROWSE_ATTEMPT_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['id'],
            'properties': {
                'id': {
                    'type': 'string',
                    'description': 'The unique identifier of the attempt to browse (e.g., "attempt-1", "attempt-2")',
                },
            },
        },
    ),
)
