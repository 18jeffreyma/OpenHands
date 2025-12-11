from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_SUBMIT_DESCRIPTION = """Submit a completed attempt as the final solution and skip remaining iterations."""

SubmitAttemptAsFinalTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='submit_attempt_as_final',
        description=_SUBMIT_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['attempt_id'],
            'properties': {
                'attempt_id': {
                    'type': 'string',
                    'description': 'ID of the attempt to apply as final.',
                },
                'message': {
                    'type': 'string',
                    'description': 'Why this attempt is final; include validation/coverage notes.',
                },
            },
        },
    ),
)


