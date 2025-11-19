from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_FINISH_BROWSING_ATTEMPT_TOOL_NAME = "finish_browsing_attempt"
_FINISH_BROWSING_ATTEMPT_DESCRIPTION = """Signals the completion of browsing previous attempts.

Use this tool when:
- You have successfully completed browsing previous attempts
- You are ready to go back to attempting to fulfill the user's request

The message should include:
- A clear descriptive summary of work done and takeaways from browsing previous attempts
- Explanation of what should be done next to fulfill the user's request
"""

FinishBrowsingAttemptTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=_FINISH_BROWSING_ATTEMPT_TOOL_NAME,
        description=_FINISH_BROWSING_ATTEMPT_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['message'],
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'Final message summarizing browsing previous attempts and next steps',
                },
            },
        },
    ),
)
