"""Static prompt and reference documentation for LLM agents.

Prompts are loaded from markdown files in this package.
"""

from importlib.resources import files

_PROMPTS_PKG = files("activecontext.prompts")


def load_prompt(name: str) -> str:
    """Load a prompt by name (without .md extension)."""
    return _PROMPTS_PKG.joinpath(f"{name}.md").read_text(encoding="utf-8")


def list_prompts() -> list[str]:
    """List available prompt names."""
    return [
        f.name[:-3]  # Remove .md extension
        for f in _PROMPTS_PKG.iterdir()
        if f.name.endswith(".md")
    ]


# Load all prompts in logical order for building combined system context
SYSTEM_PROMPT = load_prompt("system")
CONTEXT_GUIDE = load_prompt("context_guide")
DSL_REFERENCE = load_prompt("dsl_reference")
NODE_STATES = load_prompt("node_states")
CONTEXT_GRAPH = load_prompt("context_graph")
WORK_COORDINATION = load_prompt("work_coordination")
MCP_REFERENCE = load_prompt("mcp")

# Combined prompt with all reference material
FULL_SYSTEM_PROMPT = "\n\n---\n\n".join([
    SYSTEM_PROMPT,
    CONTEXT_GUIDE,
    DSL_REFERENCE,
    NODE_STATES,
    CONTEXT_GRAPH,
    WORK_COORDINATION,
    MCP_REFERENCE,
])

__all__ = [
    "load_prompt",
    "list_prompts",
    "SYSTEM_PROMPT",
    "CONTEXT_GUIDE",
    "DSL_REFERENCE",
    "NODE_STATES",
    "CONTEXT_GRAPH",
    "WORK_COORDINATION",
    "MCP_REFERENCE",
    "FULL_SYSTEM_PROMPT",
]
