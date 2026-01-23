"""Pytest fixtures for ACP protocol tests."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio

from .helpers import ACPTestClient, initialize_agent


@pytest.fixture(scope="module")
def event_loop_policy():
    """Set event loop policy for Windows."""
    if sys.platform == "win32":
        # Use WindowsProactorEventLoopPolicy for subprocess support
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.get_event_loop_policy()


@pytest.fixture(scope="module")
def event_loop(event_loop_policy):
    """Create module-scoped event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    # Close all async generators and shutdown asyncgens
    try:
        loop.run_until_complete(loop.shutdown_asyncgens())
    except RuntimeError:
        pass
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def agent_process() -> AsyncIterator[asyncio.subprocess.Process]:
    """Start agent subprocess for test module.

    Spawns python -m activecontext and yields the process.
    Terminates on cleanup.
    """
    cwd = Path(__file__).parent.parent.parent  # Project root

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "activecontext",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )

    # Give subprocess time to initialize
    await asyncio.sleep(0.5)

    yield proc

    # Cleanup
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


@pytest_asyncio.fixture(scope="module")
async def client(agent_process: asyncio.subprocess.Process) -> AsyncIterator[ACPTestClient]:
    """Create test client connected to agent.

    Module-scoped to reuse connection across tests.
    """
    if agent_process.stdin is None or agent_process.stdout is None:
        pytest.fail("Agent process missing stdin/stdout")

    test_client = ACPTestClient(
        reader=agent_process.stdout,
        writer=agent_process.stdin,
    )
    test_client.start_handler()

    yield test_client

    await test_client.close()


@pytest_asyncio.fixture(scope="module")
async def initialized_client(client: ACPTestClient) -> ACPTestClient:
    """Client that has completed initialize handshake."""
    response = await initialize_agent(client)
    assert "result" in response, f"Initialize failed: {response}"
    return client
