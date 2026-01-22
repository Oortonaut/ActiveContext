"""Tap mode - read-only traffic logging."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
import time
from pathlib import Path
from typing import IO, TYPE_CHECKING

from rich.console import Console

from acp_debug.extension.chain import ExtensionChain
from acp_debug.extension.loader import load_extensions
from acp_debug.transport.proxy import ProxyTransport
from acp_debug.transport.router import MessageRouter
from acp_debug.transport.stdio import JsonRpcMessage, StdioTransport

if TYPE_CHECKING:
    from acp_debug.config import Config

console = Console(stderr=True)


class TapRecorder:
    """Records messages to JSONL file."""

    def __init__(self, output: IO[str]) -> None:
        self.output = output

    def record(self, direction: str, msg: JsonRpcMessage) -> None:
        """Record a message with timestamp and direction."""
        record = {
            "ts": time.time(),
            "dir": direction,
            "msg": msg.to_dict(),
        }
        self.output.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.output.flush()


async def run_tap(config: Config, agent_command: str, output: Path | None) -> int:
    """Run in tap mode.

    Args:
        config: Configuration
        agent_command: Command to spawn the agent
        output: Output file for recorded traffic (stdout if None)

    Returns:
        Exit code
    """
    # Load extensions
    extensions = load_extensions(config)
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

    # Open output file
    output_file: IO[str]
    if output:
        output_file = open(output, "w", encoding="utf-8")
        if not config.quiet:
            console.print(f"[dim]Recording to: {output}[/dim]")
    else:
        # Use stderr for recording if stdout is used for protocol
        output_file = sys.stderr
        if not config.quiet:
            console.print("[dim]Recording to stderr[/dim]")

    recorder = TapRecorder(output_file)

    # Spawn agent subprocess
    if not config.quiet:
        console.print(f"[dim]Spawning agent: {agent_command}[/dim]")

    try:
        process = await asyncio.create_subprocess_exec(
            args[0],
            *args[1:],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
        )
    except Exception as e:
        console.print(f"[red]Error spawning agent: {e}[/red]")
        return 1

    # Create transports
    ide_transport = await StdioTransport.from_stdio()
    agent_transport = await StdioTransport.from_process(process)

    # Create recording wrapper
    async def record_and_forward_to_agent(msg: JsonRpcMessage) -> JsonRpcMessage | None:
        recorder.record("c2a", msg)
        await agent_transport.write_message(msg)
        return None

    async def record_and_forward_to_ide(msg: JsonRpcMessage) -> JsonRpcMessage | None:
        recorder.record("a2c", msg)
        await ide_transport.write_message(msg)
        return None

    # Create router and proxy with recording
    router = MessageRouter(chain)
    proxy = ProxyTransport(
        ide_transport=ide_transport,
        agent_transport=agent_transport,
        router=router,
    )

    # Override forward methods to include recording
    original_forward_to_agent = proxy._forward_to_agent
    original_forward_to_ide = proxy._forward_to_ide

    async def recording_forward_to_agent(msg: JsonRpcMessage) -> JsonRpcMessage | None:
        recorder.record("c2a", msg)
        return await original_forward_to_agent(msg)

    async def recording_forward_to_ide(msg: JsonRpcMessage) -> JsonRpcMessage | None:
        recorder.record("a2c", msg)
        return await original_forward_to_ide(msg)

    proxy._forward_to_agent = recording_forward_to_agent  # type: ignore
    proxy._forward_to_ide = recording_forward_to_ide  # type: ignore

    if not config.quiet:
        console.print("[green]Tap running[/green]")

    try:
        await proxy.run()
    except KeyboardInterrupt:
        pass
    finally:
        await proxy.stop()
        chain.shutdown_all()

        # Terminate agent
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()

        await ide_transport.close()
        await agent_transport.close()

        if output:
            output_file.close()

    return process.returncode or 0
