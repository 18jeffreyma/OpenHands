from dotenv import load_dotenv

load_dotenv()


from openhands.controller.agent import Agent  # noqa: E402
from openhands.agenthub import rlm_agent  # noqa: E402


def _optional_import(module: str):
    try:  # noqa: E402
        return __import__(f'openhands.agenthub.{module}', fromlist=['*'])
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        return None


codeact_agent = _optional_import('codeact_agent')
dummy_agent = _optional_import('dummy_agent')
loc_agent = _optional_import('loc_agent')
readonly_agent = _optional_import('readonly_agent')
browsing_agent = _optional_import('browsing_agent')
visualbrowsing_agent = _optional_import('visualbrowsing_agent')

__all__ = [
    'Agent',
    'codeact_agent',
    'dummy_agent',
    'browsing_agent',
    'visualbrowsing_agent',
    'readonly_agent',
    'loc_agent',
    'rlm_agent',
]
