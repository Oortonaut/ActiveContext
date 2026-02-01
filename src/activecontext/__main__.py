"""Entry point for running ActiveContext as an ACP agent.

Usage:
    python -m activecontext

    # For testing with a file:
    cat tests/fixtures/init_and_session.jsonl | python -m activecontext

This starts the ACP agent listening on stdin/stdout for JSON-RPC
messages from an ACP client (Rider, Zed, etc.).
"""

import os
import sys

from activecontext.logging import get_logger, setup_logging

log = get_logger()


def _expand_env_vars() -> None:
    """Expand ${VAR_NAME} references in environment variables.

    This allows acp.json to use:
        "env": { "OPENAI_API_KEY": "${OPENAI_API_KEY}" }

    To pull from the system environment.
    """
    import re

    pattern = re.compile(r"\$\{([^}]+)\}")

    # Iterate over a copy since we're modifying os.environ
    for key, value in list(os.environ.items()):
        if not isinstance(value, str):
            continue

        # Find all ${VAR_NAME} patterns
        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            # Get from original environment (not the modified one)
            return os.environ.get(var_name, match.group(0))

        expanded = pattern.sub(replace_var, value)
        if expanded != value:
            os.environ[key] = expanded
            log.debug("Expanded env var: %s", key)


def _patch_acp_schema() -> None:
    """Patch ACP schema to match protocol spec.

    The agent-client-protocol package has mcpServers as REQUIRED in
    NewSessionRequest, but the ACP spec says it should be OPTIONAL.
    See: docs/acp-protocol.md - mcpServers has '?' suffix indicating optional.

    Also patches McpServerStdio.env to be optional (empty list default),
    since clients commonly omit env when no environment overrides are needed.
    """
    from acp.schema import McpServerStdio, NewSessionRequest

    field = NewSessionRequest.model_fields.get("mcp_servers")
    if field and field.is_required():
        field.default = []
        NewSessionRequest.model_rebuild(force=True)
        log.debug("Patched NewSessionRequest.mcp_servers to be optional")

    env_field = McpServerStdio.model_fields.get("env")
    if env_field and env_field.is_required():
        env_field.default = []
        McpServerStdio.model_rebuild(force=True)
        log.debug("Patched McpServerStdio.env to be optional")


async def _main() -> None:
    """Async entry point for ACP-only mode."""
    import acp

    # Apply spec compliance patches before loading agent
    _patch_acp_schema()

    from activecontext.transport.acp.agent import create_agent

    log.info("Creating agent...")
    agent = create_agent()

    if agent._current_model_id:
        log.info("Agent ready, model=%s", agent._current_model_id)
    else:
        log.warning(
            "Agent ready, no LLM configured "
            "(set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or DEEPSEEK_API_KEY)"
        )

    log.info("Ready to accept ACP requests")
    await acp.run_agent(agent, use_unstable_protocol=True)


async def _main_lsp() -> None:
    """Async entry point for LSP-only mode."""
    import asyncio

    from activecontext.transport.lsp.server import LSPServer

    log.info("Creating LSP server...")

    # Create LSP server (no document sync handler for now)
    lsp_server = LSPServer()

    # Wire up stdin/stdout
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        lambda: asyncio.streams.FlowControlMixin(), sys.stdout.buffer
    )
    loop = asyncio.get_event_loop()
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

    log.info("LSP server ready")
    await lsp_server.serve(reader, writer)


async def _main_multiplexed() -> None:
    """Async entry point for multiplexed mode (ACP + LSP)."""
    import asyncio
    from typing import Any

    from activecontext.transport.multiplexer import MultiplexedTransport, ProtocolMode

    log.info("Creating multiplexed transport...")

    # Placeholder handlers for now
    async def on_acp_message(msg: dict[str, Any]) -> None:
        log.debug("Received ACP message: %s", msg.get("method"))
        # TODO: Route to ACP agent

    async def on_lsp_message(msg: dict[str, Any]) -> None:
        log.debug("Received LSP message: %s", msg.get("method"))
        # TODO: Route to LSP server

    # Create multiplexer
    mux = MultiplexedTransport(
        mode=ProtocolMode.MULTIPLEXED,
        on_acp_message=on_acp_message,
        on_lsp_message=on_lsp_message,
    )

    # Wire up stdin/stdout
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        lambda: asyncio.streams.FlowControlMixin(), sys.stdout.buffer
    )
    loop = asyncio.get_event_loop()
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

    log.info("Multiplexed transport ready")
    await mux.start(reader, writer)

    # Keep running until the multiplexer stops
    try:
        while not mux._stopped:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        await mux.stop()


def _parse_args() -> tuple[list[tuple[str, str]], str | None, str | None, int | None]:
    """Parse CLI arguments.

    Returns:
        Tuple of (roots, transport_mode, log_context, log_context_n) where:
        - roots: List of (name, path) tuples from --root flags.
        - transport_mode: "lsp", "multiplexed", or None (default ACP).
        - log_context: Directory for context dump files, or None.
        - log_context_n: Max context dump files to keep, or None.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="activecontext",
        description="ActiveContext ACP agent",
    )
    parser.add_argument(
        "--root",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Register a filesystem root: --root project /path/to/dir",
    )

    # Transport mode flags (mutually exclusive)
    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument(
        "--lsp",
        action="store_const",
        const="lsp",
        dest="transport_mode",
        help="Run in LSP-only mode (Language Server Protocol)",
    )
    transport_group.add_argument(
        "--multiplexed",
        action="store_const",
        const="multiplexed",
        dest="transport_mode",
        help="Run in multiplexed mode (ACP + LSP on shared stdio)",
    )

    # Context dump logging
    parser.add_argument(
        "--log-context",
        metavar="DIR",
        help="Directory for context dump files (context-000001.md, ...)",
    )
    parser.add_argument(
        "--log-context-n",
        type=int,
        metavar="N",
        help="Keep only the N most recent context dump files",
    )

    args, _ = parser.parse_known_args()
    roots = [(name, path) for name, path in (args.root or [])]
    return roots, args.transport_mode, args.log_context, args.log_context_n


def main() -> None:
    """Run the ActiveContext ACP agent."""
    import asyncio

    from activecontext.config import load_config

    cli_roots, transport_mode, log_context, log_context_n = _parse_args()

    # When stdin is piped (IDE or echo), silence stderr and root logger
    # to prevent tracebacks from interfering with ACP protocol
    if not sys.stdin.isatty():
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115 - intentionally kept open for process lifetime
        import logging

        logging.getLogger().addHandler(logging.NullHandler())
        logging.getLogger().setLevel(logging.CRITICAL + 1)  # Silence everything

    _expand_env_vars()
    config = load_config()
    setup_logging(config.logging)

    # CLI overrides for context dump logging (highest priority)
    if log_context:
        config.logging.context_dir = log_context
    if log_context_n is not None:
        config.logging.context_n = log_context_n

    # Store CLI roots for session creation
    if cli_roots:
        from activecontext.session.mcp_integration import set_cli_roots

        set_cli_roots(cli_roots)
        log.info("CLI roots: %s", cli_roots)

    # Determine which transport to run
    if transport_mode == "lsp":
        log.info("Starting ActiveContext in LSP-only mode...")
        asyncio.run(_main_lsp())
    elif transport_mode == "multiplexed":
        log.info("Starting ActiveContext in multiplexed mode (ACP + LSP)...")
        asyncio.run(_main_multiplexed())
    else:
        log.info("Starting ActiveContext ACP agent...")
        asyncio.run(_main())


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

    main()
