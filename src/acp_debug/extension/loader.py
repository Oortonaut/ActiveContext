"""Extension discovery and loading."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from acp_debug.extension.base import ACPExtension

if TYPE_CHECKING:
    from acp_debug.config import Config


def load_extensions(config: Config) -> list[ACPExtension]:
    """Load extensions from config.

    Loading order:
    1. Config extensions (in order)
    2. Config paths (alphabetically per directory)
    3. CLI extensions (in order) - already merged into config
    4. CLI paths (alphabetically per directory) - already merged into config

    Returns instantiated extension objects.
    """
    extensions: list[ACPExtension] = []

    # Load from explicit extension files
    for ext_path in config.extensions:
        ext_path = Path(ext_path).resolve()
        if ext_path.exists():
            loaded = _load_extensions_from_file(ext_path)
            extensions.extend(loaded)

    # Load from extension directories
    for dir_path in config.extensions_path:
        dir_path = Path(dir_path).resolve()
        if dir_path.is_dir():
            # Load .py files alphabetically
            for py_file in sorted(dir_path.glob("*.py")):
                if not py_file.name.startswith("_"):
                    loaded = _load_extensions_from_file(py_file)
                    extensions.extend(loaded)

    return extensions


def _load_extensions_from_file(path: Path) -> list[ACPExtension]:
    """Load ACPExtension subclasses from a Python file."""
    extensions: list[ACPExtension] = []

    # Generate unique module name
    module_name = f"acp_debug_ext_{path.stem}_{id(path)}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return extensions

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find ACPExtension subclasses (but not ACPExtension itself or base classes)
        from acp_debug.extension.mock_agent import MockAgentBase
        from acp_debug.extension.mock_client import MockClientBase

        base_classes = {ACPExtension, MockAgentBase, MockClientBase}

        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, ACPExtension)
                and obj not in base_classes
            ):
                # Instantiate the extension
                instance = obj()
                extensions.append(instance)

    except Exception as e:
        # Log but don't fail - allow other extensions to load
        print(f"Warning: Failed to load extension from {path}: {e}", file=sys.stderr)

    return extensions


def reload_extensions(
    config: Config, current_extensions: list[ACPExtension]
) -> list[ACPExtension]:
    """Reload extensions from configured paths.

    Calls on_shutdown() on current extensions before replacing them.
    """
    # Shutdown current extensions
    for ext in current_extensions:
        try:
            ext.on_shutdown()
        except Exception:
            pass

    # Clear any cached modules
    modules_to_remove = [
        name for name in sys.modules if name.startswith("acp_debug_ext_")
    ]
    for name in modules_to_remove:
        del sys.modules[name]

    # Load fresh
    return load_extensions(config)
