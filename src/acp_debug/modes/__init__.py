"""Operating modes for acp-debug."""

from acp_debug.modes.mock_agent import run_mock_agent
from acp_debug.modes.mock_client import run_mock_client
from acp_debug.modes.proxy import run_proxy
from acp_debug.modes.tap import run_tap

__all__ = [
    "run_proxy",
    "run_mock_agent",
    "run_mock_client",
    "run_tap",
]
