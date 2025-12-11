from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_BROWSE_DESCRIPTION = """Browse summaries of previous attempts to decide what to expand.

Call this at the start of REFLECT to see the list of attempts. Use expand_previous_attempt for details.
"""

BrowsePreviousAttemptsTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='browse_previous_attempts',
        description=_BROWSE_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'attempt_ids': {
                    'type': 'array',
                    'description': 'Optional list of attempt IDs to focus on.',
                    'items': {'type': 'string'},
                },
            },
        },
    ),
)


