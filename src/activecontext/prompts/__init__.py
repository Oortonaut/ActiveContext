"""Compatibility shim - delegates to activecontext.resources."""

from activecontext.resources import list_prompts, load_prompt

SYSTEM_PROMPT = load_prompt("system")
CONTEXT_GUIDE = load_prompt("context_guide")
DSL_REFERENCE = load_prompt("dsl_reference")
NODE_STATES = load_prompt("node_states")
CONTEXT_GRAPH = load_prompt("context_graph")
WORK_COORDINATION = load_prompt("work_coordination")
MCP_REFERENCE = load_prompt("mcp")

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
]
