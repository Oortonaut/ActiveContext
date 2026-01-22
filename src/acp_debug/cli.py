"""Command-line interface for acp-debug."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="acp-debug",
        description="ACP Protocol Debugger - Debug Agent Client Protocol interactions",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config file path (default: ./acp-debug.yaml)",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        type=Path,
        default=[],
        help="Extension files to load",
    )
    parser.add_argument(
        "--extensions-path",
        nargs="*",
        type=Path,
        default=[],
        help="Directories to scan for extensions",
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Proxy mode
    proxy_parser = subparsers.add_parser(
        "proxy",
        help="MITM proxy between IDE and agent",
    )
    proxy_parser.add_argument(
        "--agent",
        required=True,
        help="Agent command to spawn (e.g., 'python -m activecontext')",
    )

    # Mock agent mode
    mock_agent_parser = subparsers.add_parser(
        "mock-agent",
        help="Mock agent that responds to IDE",
    )
    mock_agent_parser.add_argument(
        "--script",
        type=Path,
        help="Script defining mock responses",
    )

    # Mock client mode
    mock_client_parser = subparsers.add_parser(
        "mock-client",
        help="Mock client that drives an agent",
    )
    mock_client_parser.add_argument(
        "--agent",
        required=True,
        help="Agent command to spawn",
    )
    mock_client_parser.add_argument(
        "--script",
        type=Path,
        help="Script defining requests to send",
    )

    # Tap mode
    tap_parser = subparsers.add_parser(
        "tap",
        help="Read-only traffic tap",
    )
    tap_parser.add_argument(
        "--agent",
        required=True,
        help="Agent command to spawn",
    )
    tap_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for recorded traffic (JSONL)",
    )

    return parser


def run_cli(args: Sequence[str]) -> int:
    """Run the CLI with the given arguments."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    if parsed.mode is None:
        parser.print_help()
        return 1

    # Load config
    from acp_debug.config import load_config
    config = load_config(
        config_path=parsed.config,
        extensions=parsed.extensions,
        extensions_path=parsed.extensions_path,
    )

    # Dispatch to mode
    if parsed.mode == "proxy":
        from acp_debug.modes.proxy import run_proxy
        return asyncio.run(run_proxy(config, parsed.agent))
    elif parsed.mode == "mock-agent":
        from acp_debug.modes.mock_agent import run_mock_agent
        return asyncio.run(run_mock_agent(config, parsed.script))
    elif parsed.mode == "mock-client":
        from acp_debug.modes.mock_client import run_mock_client
        return asyncio.run(run_mock_client(config, parsed.agent, parsed.script))
    elif parsed.mode == "tap":
        from acp_debug.modes.tap import run_tap
        return asyncio.run(run_tap(config, parsed.agent, parsed.output))
    else:
        parser.print_help()
        return 1
