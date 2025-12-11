from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_FINISH_REFLECTION_DESCRIPTION = """Conclude the REFLECT phase with a plan for the next attempt.

Use this after reviewing previous attempts. Include:
- A concrete plan for the next attempt or rollout steps
- Key risks or checks to perform
"""

FinishReflectionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish_reflection',
        description=_FINISH_REFLECTION_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['plan'],
            'properties': {
                'plan': {
                    'type': 'string',
                    'description': 'Actionable plan for the next attempt.',
                },
                'risks': {
                    'type': 'string',
                    'description': 'Risks, unknowns, or mitigations for the next attempt.',
                },
            },
        },
    ),
)


