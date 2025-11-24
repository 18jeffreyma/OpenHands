from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_SUBMIT_ATTEMPT_AS_FINAL_TOOL_NAME = "submit_attempt_as_final"
_SUBMIT_ATTEMPT_AS_FINAL_DESCRIPTION = """Submit a specific attempt as the final solution instead of continuing with more attempts.

Use this tool when:
- You have reviewed previous attempts and found one that successfully solves the task
- You want to submit a specific attempt as the final answer instead of making more attempts
- You are confident that a previous attempt is the best solution

This will immediately finish the task with the selected attempt, skipping any remaining iterations.
"""

SubmitAttemptAsFinalTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=_SUBMIT_ATTEMPT_AS_FINAL_TOOL_NAME,
        description=_SUBMIT_ATTEMPT_AS_FINAL_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['attempt_id', 'message'],
            'properties': {
                'attempt_id': {
                    'type': 'string',
                    'description': 'The unique identifier of the attempt to submit as final (e.g., "attempt-1", "attempt-2")',
                },
                'message': {
                    'type': 'string',
                    'description': 'Explanation of why this attempt is being submitted as the final solution',
                },
            },
        },
    ),
)

