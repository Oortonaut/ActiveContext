"""Proxy mode - MITM between IDE and agent."""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
import signal
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

# Windows-specific subprocess creation flags
_WINDOWS = platform.system() == "Windows"
_CREATE_NEW_PROCESS_GROUP = 0x00000200 if _WINDOWS else 0


def _send_interrupt(process: asyncio.subprocess.Process) -> None:
    """Send interrupt signal to process (Ctrl-C on Windows, SIGINT on Unix)."""
    if _WINDOWS:
        # On Windows, send CTRL_BREAK_EVENT to the process group
        # CTRL_C_EVENT doesn't work reliably for subprocesses
        try:
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        except OSError:
            process.terminate()
    else:
        # On Unix, send SIGINT
        try:
            os.kill(process.pid, signal.SIGINT)
        except OSError:
            process.terminate()


def _send_terminate(process: asyncio.subprocess.Process) -> None:
    """Send terminate signal to process (SIGTERM on Unix, TerminateProcess on Windows)."""
    if _WINDOWS:
        # On Windows, terminate() calls TerminateProcess
        process.terminate()
    else:
        # On Unix, send SIGTERM
        try:
            os.kill(process.pid, signal.SIGTERM)
        except OSError:
            process.terminate()


async def graceful_shutdown(
    process: asyncio.subprocess.Process,
    interrupt_timeout: float = 2.0,
    terminate_timeout: float = 3.0,
) -> None:
    """Gracefully shutdown a process: interrupt → terminate → kill.

    Args:
        process: The subprocess to shutdown
        interrupt_timeout: Seconds to wait after sending interrupt signal
        terminate_timeout: Seconds to wait after sending terminate signal
    """
    if process.returncode is not None:
        return

    # First try interrupt (Ctrl-C/SIGINT)
    _send_interrupt(process)
    try:
        await asyncio.wait_for(process.wait(), timeout=interrupt_timeout)
        return
    except asyncio.TimeoutError:
        pass

    # Then try terminate (SIGTERM)
    _send_terminate(process)
    try:
        await asyncio.wait_for(process.wait(), timeout=terminate_timeout)
        return
    except asyncio.TimeoutError:
        pass

    # Finally force kill
    process.kill()
    await process.wait()


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

        await graceful_shutdown(
            process,
            interrupt_timeout=config.shutdown.interrupt_timeout,
            terminate_timeout=config.shutdown.terminate_timeout,
        )

        await ide_transport.close()
        await agent_transport.close()

    return process.returncode or 0
