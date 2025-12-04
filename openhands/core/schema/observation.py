from enum import Enum


class ObservationType(str, Enum):
    READ = 'read'
    """The content of a file
    """

    WRITE = 'write'

    EDIT = 'edit'

    BROWSE = 'browse'
    """The HTML content of a URL
    """

    RUN = 'run'
    """The output of a command
    """

    RUN_IPYTHON = 'run_ipython'
    """Runs a IPython cell.
    """

    CHAT = 'chat'
    """A message from the user
    """

    DELEGATE = 'delegate'
    """The result of a task delegated to another agent
    """

    MESSAGE = 'message'

    ERROR = 'error'

    SUCCESS = 'success'

    NULL = 'null'

    THINK = 'think'

    AGENT_STATE_CHANGED = 'agent_state_changed'

    USER_REJECTED = 'user_rejected'

    CONDENSE = 'condense'
    """Result of a condensation operation."""

    RECALL = 'recall'
    """Result of a recall operation. This can be the workspace context, a microagent, or other types of information."""

    MCP = 'mcp'
    """Result of a MCP Server operation"""

    DOWNLOAD = 'download'
    """Result of downloading/opening a file via the browser"""

    TASK_TRACKING = 'task_tracking'
    """Result of a task tracking operation"""

    LOOP_DETECTION = 'loop_detection'
    """Results of a dead-loop detection"""

    FINISH_ATTEMPT = 'finish_attempt'
    """Result of finishing an attempt"""

    BROWSE_PREVIOUS_ATTEMPTS = 'browse_previous_attempts'
    """Result of browsing previous attempts"""

    EXPAND_PREVIOUS_ATTEMPT = 'expand_previous_attempt'
    """Result of expanding a previous attempt"""

    FINISH_REFLECTION = 'finish_reflection'
    """Result of finishing the reflection phase"""

    FINISH_CHARACTERIZATION = 'finish_characterization'
    """Result of finishing the characterization phase"""

    SUBMIT_ATTEMPT_AS_FINAL = 'submit_attempt_as_final'
    """Result of submitting an attempt as the final solution"""
