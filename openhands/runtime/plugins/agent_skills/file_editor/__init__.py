"""This file imports a global singleton of the `EditTool` class as well as raw functions that expose
its __call__.
The implementation of the `EditTool` class can be found at: https://github.com/OpenHands/openhands-aci/.
"""

try:
    from openhands_aci.editor import file_editor
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    file_editor = None

__all__ = ['file_editor']
