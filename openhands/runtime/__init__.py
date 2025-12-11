import importlib

from openhands.runtime.base import Runtime
from openhands.utils.import_utils import get_impl

try:
    from openhands.runtime.impl.cli.cli_runtime import CLIRuntime
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    CLIRuntime = None
try:
    from openhands.runtime.impl.docker.docker_runtime import DockerRuntime
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    DockerRuntime = None
try:
    from openhands.runtime.impl.kubernetes.kubernetes_runtime import KubernetesRuntime
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    KubernetesRuntime = None
try:
    from openhands.runtime.impl.local.local_runtime import LocalRuntime
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    LocalRuntime = None
try:
    from openhands.runtime.impl.remote.remote_runtime import RemoteRuntime
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    RemoteRuntime = None

# mypy: disable-error-code="type-abstract"
_DEFAULT_RUNTIME_CLASSES: dict[str, type[Runtime]] = {}
if DockerRuntime:
    _DEFAULT_RUNTIME_CLASSES.update({'eventstream': DockerRuntime, 'docker': DockerRuntime})
if RemoteRuntime:
    _DEFAULT_RUNTIME_CLASSES['remote'] = RemoteRuntime
if LocalRuntime:
    _DEFAULT_RUNTIME_CLASSES['local'] = LocalRuntime
if KubernetesRuntime:
    _DEFAULT_RUNTIME_CLASSES['kubernetes'] = KubernetesRuntime
if CLIRuntime:
    _DEFAULT_RUNTIME_CLASSES['cli'] = CLIRuntime

# Try to import third-party runtimes if available
_THIRD_PARTY_RUNTIME_CLASSES: dict[str, type[Runtime]] = {}

# Dynamically discover and import third-party runtimes

# Check if third_party package exists and discover runtimes
try:
    import third_party.runtime.impl

    third_party_base = 'third_party.runtime.impl'

    # List of potential third-party runtime modules to try
    # These are discovered from the third_party directory structure
    potential_runtimes = []
    try:
        import pkgutil

        for importer, modname, ispkg in pkgutil.iter_modules(
            third_party.runtime.impl.__path__
        ):
            if ispkg:
                potential_runtimes.append(modname)
    except Exception:
        # If discovery fails, no third-party runtimes will be loaded
        potential_runtimes = []

    # Try to import each discovered runtime
    for runtime_name in potential_runtimes:
        try:
            module_path = f'{third_party_base}.{runtime_name}.{runtime_name}_runtime'
            module = importlib.import_module(module_path)

            # Try different class name patterns
            possible_class_names = [
                f'{runtime_name.upper()}Runtime',  # E2BRuntime
                f'{runtime_name.capitalize()}Runtime',  # E2bRuntime, DaytonaRuntime, etc.
            ]

            runtime_class = None
            for class_name in possible_class_names:
                try:
                    runtime_class = getattr(module, class_name)
                    break
                except AttributeError:
                    continue

            if runtime_class:
                _THIRD_PARTY_RUNTIME_CLASSES[runtime_name] = runtime_class

        except ImportError:
            # ImportError means the library is not installed (expected for optional dependencies)
            pass
        except Exception as e:
            # Other exceptions mean the library is present but broken, which should be logged
            from openhands.core.logger import openhands_logger as logger

            logger.warning(f'Failed to import third-party runtime {module_path}: {e}')
            pass

except ImportError:
    # third_party package not available
    pass

# Combine core and third-party runtimes
_ALL_RUNTIME_CLASSES = {**_DEFAULT_RUNTIME_CLASSES, **_THIRD_PARTY_RUNTIME_CLASSES}


def get_runtime_cls(name: str) -> type[Runtime]:
    """If name is one of the predefined runtime names (e.g. 'docker'), return its class.
    Otherwise attempt to resolve name as subclass of Runtime and return it.
    Raise on invalid selections.
    """
    if name in _ALL_RUNTIME_CLASSES:
        return _ALL_RUNTIME_CLASSES[name]
    try:
        return get_impl(Runtime, name)
    except Exception as e:
        known_keys = _ALL_RUNTIME_CLASSES.keys()
        raise ValueError(
            f'Runtime {name} not supported, known are: {known_keys}'
        ) from e


# Build __all__ list dynamically based on available runtimes
__all__ = ['Runtime', 'get_runtime_cls']
for runtime_cls in (RemoteRuntime, DockerRuntime, KubernetesRuntime, CLIRuntime, LocalRuntime):
    if runtime_cls:
        __all__.append(runtime_cls.__name__)

# Add third-party runtimes to __all__ if they're available
for runtime_name, runtime_class in _THIRD_PARTY_RUNTIME_CLASSES.items():
    __all__.append(runtime_class.__name__)
