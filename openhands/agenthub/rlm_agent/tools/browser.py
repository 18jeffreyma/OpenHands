try:
    from browsergym.core.action.highlevel import HighLevelActionSet
    from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk

    from openhands.agenthub.rlm_agent.tools.security_utils import (
        RISK_LEVELS,
        SECURITY_RISK_DESC,
    )
    from openhands.llm.tool_names import BROWSER_TOOL_NAME
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    BrowserTool = None
else:
    _browser_action_space = HighLevelActionSet(
        subsets=['bid', 'nav'],
        strict=False,
        multiaction=True,
    )

    _BROWSER_DESCRIPTION = """Interact with the browser using Python code. Use it ONLY when you need to interact with a webpage.

Multiple actions can be provided at once, but will be executed sequentially without any feedback from the page.
You can also use the browser to view pdf, png, jpg files by opening http://localhost:8000/view?path=<absolute_path>.
"""

    _BROWSER_TOOL_DESCRIPTION = """
The following functions are available. Nothing else is supported.

goto(url: str)
go_back()
go_forward()
noop(wait_ms: float = 1000)
scroll(delta_x: float, delta_y: float)
fill(bid: str, value: str)
select_option(bid: str, options: str | list[str])
click(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
dblclick(bid: str, button: Literal['left', 'middle', 'right'] = 'left', modifiers: list[typing.Literal['Alt', 'Control', 'ControlOrMeta', 'Meta', 'Shift']] = [])
hover(bid: str)
press(bid: str, key_comb: str)
focus(bid: str)
clear(bid: str)
upload_file(bid: str, file_path: str)
switch_tab(tab: str, create: bool = False)
close_tab(tab: str)
activate_chrome_extension(extension_name: str)
"""

    BrowserTool = ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name=BROWSER_TOOL_NAME,
            description=_BROWSER_DESCRIPTION,
            parameters={
                'type': 'object',
                'required': ['code'],
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': _BROWSER_TOOL_DESCRIPTION,
                    },
                    'security_risk': {
                        'type': 'string',
                        'description': SECURITY_RISK_DESC,
                        'enum': RISK_LEVELS,
                    },
                },
            },
        ),
    )


