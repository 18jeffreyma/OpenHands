from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

from openhands.llm.tool_names import FINISH_TOOL_NAME

_FINISH_DESCRIPTION = """Signals the completion of the current ATTEMPT.

Use this tool when:
- You have made code edits and observed their effect (success OR failure)
- You have run a test and seen the results (regardless of outcome)
- You encounter an error - report it and finish, do NOT try to fix it

CRITICAL: After seeing test results (especially failures or regressions), you MUST call finish IMMEDIATELY. Do NOT:
- Undo or revert your changes
- Try to fix or improve your changes
- Plan next steps or future attempts
- Investigate further or look for alternative approaches

Just call finish and report what happened. The REFLECT phase will decide what to do next.

The message should include:
- A clear summary of actions taken and their results
- Any observations about the results
- Explanation if your approach didn't work as expected
- Any opportunities for future attempts

"""

FinishTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=FINISH_TOOL_NAME,
        description=_FINISH_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['message'],
            'properties': {
                'message': {
                    'type': 'string',
                    'description': 'Final message to send to the user',
                },
            },
        },
    ),
)
