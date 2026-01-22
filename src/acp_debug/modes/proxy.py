"""Proxy mode - MITM between IDE and agent."""

from __future__ import annotations

import asyncio
import shlex
import sys
from typing import TYPE_CHECKING

from rich.console import Console

from acp_debug.extension.chain import ExtensionChain
from acp_debug.extension.loader import load_extensions
from acp_debug.transport.proxy import ProxyTransport
from acp_debug.transport.router import MessageRouter
from acp_debug.transport.stdio import StdioTransport

if TYPE_CHECKING:
    from acp_debug.config import Config

console = Console(stderr=True)


async def run_proxy(config: Config, agent_command: str) -> int:
    """Run in proxy mode.

    Args:
        config: Configuration
        agent_command: Command to spawn the agent

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

    # Create router and proxy
    router = MessageRouter(chain)
    proxy = ProxyTransport(
        ide_transport=ide_transport,
        agent_transport=agent_transport,
        router=router,
    )

    if not config.quiet:
        console.print("[green]Proxy running[/green]")

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

    return process.returncode or 0
