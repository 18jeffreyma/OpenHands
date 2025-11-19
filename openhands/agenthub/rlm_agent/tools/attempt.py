from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_ATTEMPT_TOOL_NAME = "finish_attempt"
_ATTEMPT_DESCRIPTION = """Signals the completion of an attempt to fulfill the user's request.

Use this tool when:
- You have successfully completed the user's requested task
- You cannot proceed further due to technical limitations or missing information

The message should include:
- A clear descriptive summary of work done during the attempt, actions taken and their results
- Explanation if you're unable to complete the task
- Suggestions for next steps or alternative approaches if applicable
"""

FinishAttemptTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=_ATTEMPT_TOOL_NAME,
        description=_ATTEMPT_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['message'],
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'Final message summarizing the attempt',
                },
            },
        },
    ),
)
