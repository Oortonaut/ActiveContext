"""Extension system for acp-debug."""

from acp_debug.extension.base import ACPExtension
from acp_debug.extension.chain import ExtensionChain
from acp_debug.extension.loader import load_extensions
from acp_debug.extension.mock_agent import MockAgentBase
from acp_debug.extension.mock_client import MockClientBase

__all__ = [
    "ACPExtension",
    "ExtensionChain",
    "load_extensions",
    "MockAgentBase",
    "MockClientBase",
]
