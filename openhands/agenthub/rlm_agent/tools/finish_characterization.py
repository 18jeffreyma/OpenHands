from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

_FINISH_CHARACTERIZATION_DESCRIPTION = """Summarize and label the attempt after analysis.

Use this when you have finished CHARACTERIZE. Include:
- A short title/label for the attempt
- A semantic summary of what changed and why
- Validation results (tests/lint/perf) and confidence
- Limitations or follow-ups
"""

FinishCharacterizationTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='finish_characterization',
        description=_FINISH_CHARACTERIZATION_DESCRIPTION,
        parameters={
            'type': 'object',
            'required': ['title', 'summary'],
            'properties': {
                'title': {
                    'type': 'string',
                    'description': 'Short title/label for this attempt characterization.',
                },
                'summary': {
                    'type': 'string',
                    'description': 'Semantic summary of the attempt and changes made.',
                },
                'validation_results': {
                    'type': 'string',
                    'description': 'Results from tests/lint/perf or manual checks.',
                },
                'confidence': {
                    'type': 'string',
                    'description': 'Confidence level: high, medium, or low.',
                    'enum': ['high', 'medium', 'low'],
                },
                'limitations': {
                    'type': 'string',
                    'description': 'Known gaps, risks, or follow-up items.',
                },
            },
        },
    ),
)


