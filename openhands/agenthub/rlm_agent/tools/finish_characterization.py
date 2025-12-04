from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_FINISH_CHARACTERIZATION_TOOL_NAME = "finish_characterization"
_FINISH_CHARACTERIZATION_DESCRIPTION = """Signals the completion of the characterization phase for the current attempt.

Use this tool after you have:
- Analyzed the changes made during the attempt
- Run any necessary profiling or testing commands
- Generated a comprehensive semantic summary of the attempt

The characterization_summary should be a rich, descriptive summary that includes:
- A concise title/label for this attempt (2-5 words)
- Key modifications made (files changed, functions modified, etc.)
- Performance characteristics (if applicable - runtime, memory usage, etc.)
- Test results or validation outcomes (if applicable)
- Confidence level in the solution (high/medium/low)
- Known limitations or edge cases
- Comparison to previous attempts (if any)

This summary will be used to identify and compare attempts during the reflection phase.
"""

FinishCharacterizationTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name=_FINISH_CHARACTERIZATION_TOOL_NAME,
        description=_FINISH_CHARACTERIZATION_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['characterization_summary'],
            'properties': {
                'characterization_summary': {
                    'type': 'string',
                    'description': 'Semantic summary characterizing this attempt',
                },
            },
        },
    ),
)

