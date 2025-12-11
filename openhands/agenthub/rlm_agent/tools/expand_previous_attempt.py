from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_EXPAND_DESCRIPTION = """Expand a specific attempt with more detail (actions, patches, notes).

Provide the attempt_id you want to inspect.
"""

ExpandPreviousAttemptTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='expand_previous_attempt',
        description=_EXPAND_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['attempt_id'],
            'properties': {
                'attempt_id': {
                    'type': 'string',
                    'description': 'ID of the attempt to expand.',
                },
            },
        },
    ),
)


