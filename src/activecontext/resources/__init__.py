"""Package resources for ActiveContext.

Consolidates all bundled resources (prompts, configs) into one location.
Use load_resource() for arbitrary paths, load_prompt() for prompts.
"""

from __future__ import annotations

from importlib.resources import files

_RESOURCES_PKG = files("activecontext.resources")


def load_resource(path: str) -> str:
    """Load a text resource by relative path (e.g. "prompts/system.md")."""
    return _RESOURCES_PKG.joinpath(path).read_text(encoding="utf-8")


def load_prompt(name: str) -> str:
    """Load a prompt by name. Supports paths like "modes/normal".

    Args:
        name: Prompt name with optional subdirectory, with or without .md extension.
    """
    if not name.endswith(".md"):
        name = f"{name}.md"
    return load_resource(f"prompts/{name}")


def list_prompts() -> list[str]:
    """List available top-level prompt names (without .md extension)."""
    prompts_dir = _RESOURCES_PKG.joinpath("prompts")
    return [f.name[:-3] for f in prompts_dir.iterdir() if f.name.endswith(".md")]
