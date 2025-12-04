from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_FINISH_REFLECTION_TOOL_NAME = "finish_reflection"
_FINISH_REFLECTION_DESCRIPTION = """Signals the completion of the reflection phase.

Use this tool when:
- You have successfully completed reviewing and analyzing previous attempts
- You have formed a clear plan for the next attempt (or identified a successful attempt to submit)

The message should include:
- Key insights from reviewing previous attempts
- What worked well and what didn't
- A detailed plan for the next attempt phase
- Specific steps to take and why they should work better
"""

FinishReflectionTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=_FINISH_REFLECTION_TOOL_NAME,
        description=_FINISH_REFLECTION_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['message'],
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'Summary of insights from reflection and plan for next attempt',
                },
            },
        },
    ),
)

