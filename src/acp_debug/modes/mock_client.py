"""Mock client mode - drive an agent with scripted requests."""

from __future__ import annotations

import asyncio
import platform
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from acp_debug.extension.chain import ExtensionChain
from acp_debug.extension.loader import load_extensions
from acp_debug.extension.mock_client import MockClientBase
from acp_debug.modes.proxy import graceful_shutdown
from acp_debug.transport.stdio import JsonRpcMessage, StdioTransport

if TYPE_CHECKING:
    from acp_debug.config import Config

console = Console(stderr=True)

# Windows-specific subprocess creation flags
_WINDOWS = platform.system() == "Windows"
_CREATE_NEW_PROCESS_GROUP = 0x00000200 if _WINDOWS else 0


async def run_mock_client(config: Config, agent_command: str, script: Path | None) -> int:
    """Run in mock client mode.

    Args:
        config: Configuration
        agent_command: Command to spawn the agent
        script: Optional script defining requests to send

    Returns:
        Exit code
    """
    # Load extensions
    extensions = load_extensions(config)

    # Find mock client extension or use default
    mock_client: MockClientBase | None = None
    for ext in extensions:
        if isinstance(ext, MockClientBase):
            mock_client = ext
            break

    if mock_client is None:
        mock_client = MockClientBase()
        extensions.insert(0, mock_client)

    if not config.quiet:
        console.print(f"[dim]Loaded {len(extensions)} extension(s)[/dim]")

    # Create extension chain
    chain = ExtensionChain(extensions)
    chain.initialize_all()

    # Parse agent command
    args = shlex.split(agent_command)
    if not args:
        console.print("[red]Error: Empty agent command[/red]")
        return 1

    # Spawn agent subprocess
    if not config.quiet:
        console.print(f"[dim]Spawning agent: {agent_command}[/dim]")

    try:
        # On Windows, create in new process group to enable Ctrl+Break signaling
        creationflags = _CREATE_NEW_PROCESS_GROUP if _WINDOWS else 0
        process = await asyncio.create_subprocess_exec(
            args[0],
            *args[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            creationflags=creationflags,  # type: ignore[arg-type]
        )
    except Exception as e:
        console.print(f"[red]Error spawning agent: {e}[/red]")
        return 1

    # Create transport
    transport = await StdioTransport.from_process(process)

    if not config.quiet:
        console.print("[green]Mock client running[/green]")

    # Track pending requests
    pending: dict[int | str, asyncio.Future[JsonRpcMessage]] = {}
    next_id = 0

    async def send_request(method: str, params: dict) -> JsonRpcMessage:
        """Send a request to the agent and wait for response."""
        nonlocal next_id
        msg_id = next_id
        next_id += 1

        future: asyncio.Future[JsonRpcMessage] = asyncio.Future()
        pending[msg_id] = future

        msg = JsonRpcMessage(id=msg_id, method=method, params=params)
        await transport.write_message(msg)

        return await future

    async def handle_agent_messages():
        """Handle messages from agent."""
        async for msg in transport.messages():
            if msg.is_response() and msg.id in pending:
                pending[msg.id].set_result(msg)
            elif msg.is_request():
                # Agent requesting something from us (e.g., permission)
                response = await _handle_agent_request(chain, mock_client, msg)
                if response:
                    await transport.write_message(response)
            elif msg.is_notification():
                # Log notifications
                if not config.quiet:
                    console.print(f"[dim]← {msg.method}[/dim]")

    # Start message handler
    handler_task = asyncio.create_task(handle_agent_messages())

    try:
        # If script provided, run it
        if script:
            # TODO: Load and execute script
            console.print(f"[yellow]Script execution not yet implemented: {script}[/yellow]")
        else:
            # Interactive mode - basic example
            # Initialize
            console.print("[dim]Sending initialize...[/dim]")
            response = await send_request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {"readTextFile": True, "writeTextFile": True},
                        "terminal": True,
                    },
                    "clientInfo": {"name": "acp-debug", "title": "ACP Debug", "version": "0.1.0"},
                },
            )
            console.print(f"[green]Agent initialized: {response.result}[/green]")

            # Create session
            console.print("[dim]Creating session...[/dim]")
            response = await send_request(
                "session/new",
                {"cwd": str(Path.cwd())},
            )
            session_id = response.result.get("sessionId") if response.result else None
            console.print(f"[green]Session created: {session_id}[/green]")

            # Wait for user to exit
            console.print("\n[dim]Press Ctrl+C to exit[/dim]")
            await asyncio.Event().wait()

    except KeyboardInterrupt:
        pass
    finally:
        handler_task.cancel()
        try:
            await handler_task
        except asyncio.CancelledError:
            pass

        chain.shutdown_all()

        await graceful_shutdown(
            process,
            interrupt_timeout=config.shutdown.interrupt_timeout,
            terminate_timeout=config.shutdown.terminate_timeout,
        )

        await transport.close()

    return process.returncode or 0


async def _handle_agent_request(
    chain: ExtensionChain,
    mock: MockClientBase,
    msg: JsonRpcMessage,
) -> JsonRpcMessage | None:
    """Handle a request from the agent using mock client."""
    from acp_debug.transport.router import CLIENT_METHODS, METHOD_TO_CHAIN

    method = msg.method
    if method is None:
        return None

    if method in CLIENT_METHODS:
        req_type, resp_type = CLIENT_METHODS[method]
        chain_method = METHOD_TO_CHAIN[method]

        req = req_type.model_validate(msg.params or {})

        # For mock client, the "final" calls the mock's method directly
        mock_method = getattr(mock, chain_method)

        async def final(r):
            return await mock_method(r, _noop)

        chain_fn = getattr(chain, chain_method)
        result = await chain_fn(req, final)

        if result is not None:
            return JsonRpcMessage(
                id=msg.id,
                result=result.model_dump(by_alias=True, exclude_none=True),
            )
        return JsonRpcMessage(id=msg.id, result=None)

    return JsonRpcMessage(
        id=msg.id,
        error={"code": -32601, "message": f"Method not found: {method}"},
    )


async def _noop(x):
    """No-op callable for mock methods."""
    return x
