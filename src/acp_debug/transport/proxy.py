"""Proxy transport - bidirectional stdio between IDE and agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from acp_debug.transport.router import MessageRouter
from acp_debug.transport.stdio import JsonRpcMessage, StdioTransport


@dataclass
class ProxyTransport:
    """Bidirectional proxy between IDE (stdin/stdout) and agent (subprocess).

    Messages flow:
    - IDE → (stdin) → Proxy → (subprocess stdin) → Agent
    - Agent → (subprocess stdout) → Proxy → (stdout) → IDE

    Extensions intercept both directions.
    """

    ide_transport: StdioTransport
    agent_transport: StdioTransport
    router: MessageRouter
    _pending_requests: dict[int | str, asyncio.Future[JsonRpcMessage]] = field(
        default_factory=dict
    )
    _next_id: int = 0
    _running: bool = False

    async def run(self) -> None:
        """Run the proxy, forwarding messages in both directions."""
        self._running = True

        # Run both direction handlers concurrently
        await asyncio.gather(
            self._handle_ide_to_agent(),
            self._handle_agent_to_ide(),
            return_exceptions=True,
        )

    async def stop(self) -> None:
        """Stop the proxy."""
        self._running = False

    async def _handle_ide_to_agent(self) -> None:
        """Handle messages from IDE to agent."""
        async for msg in self.ide_transport.messages():
            if not self._running:
                break

            if msg.is_request():
                # Route request through extensions
                response = await self.router.route_to_agent(
                    msg, self._forward_to_agent_and_wait
                )
                if response:
                    await self.ide_transport.write_message(response)

            elif msg.is_notification():
                # Route notification through extensions
                await self.router.route_to_agent(msg, self._forward_to_agent)

            elif msg.is_response():
                # Response from IDE to a request we made - resolve pending
                if msg.id is not None and msg.id in self._pending_requests:
                    self._pending_requests[msg.id].set_result(msg)

    async def _handle_agent_to_ide(self) -> None:
        """Handle messages from agent to IDE."""
        async for msg in self.agent_transport.messages():
            if not self._running:
                break

            if msg.is_request():
                # Agent is requesting something from IDE (e.g., permission)
                response = await self.router.route_to_client(
                    msg, self._forward_to_ide_and_wait
                )
                if response:
                    await self.agent_transport.write_message(response)

            elif msg.is_notification():
                # Agent sending notification to IDE (e.g., session/update)
                await self.router.route_to_client(msg, self._forward_to_ide)

            elif msg.is_response():
                # Response from agent to a request we forwarded - resolve pending
                if msg.id is not None and msg.id in self._pending_requests:
                    self._pending_requests[msg.id].set_result(msg)

    async def _forward_to_agent(self, msg: JsonRpcMessage) -> JsonRpcMessage | None:
        """Forward a message to the agent (no response expected)."""
        await self.agent_transport.write_message(msg)
        return None

    async def _forward_to_agent_and_wait(self, msg: JsonRpcMessage) -> JsonRpcMessage | None:
        """Forward a request to agent and wait for response."""
        if msg.id is None:
            await self.agent_transport.write_message(msg)
            return None

        # Create future for response
        future: asyncio.Future[JsonRpcMessage] = asyncio.Future()
        self._pending_requests[msg.id] = future

        try:
            await self.agent_transport.write_message(msg)
            return await future
        finally:
            self._pending_requests.pop(msg.id, None)

    async def _forward_to_ide(self, msg: JsonRpcMessage) -> JsonRpcMessage | None:
        """Forward a message to the IDE (no response expected)."""
        await self.ide_transport.write_message(msg)
        return None

    async def _forward_to_ide_and_wait(self, msg: JsonRpcMessage) -> JsonRpcMessage | None:
        """Forward a request to IDE and wait for response."""
        if msg.id is None:
            await self.ide_transport.write_message(msg)
            return None

        # Create future for response
        future: asyncio.Future[JsonRpcMessage] = asyncio.Future()
        self._pending_requests[msg.id] = future

        try:
            await self.ide_transport.write_message(msg)
            return await future
        finally:
            self._pending_requests.pop(msg.id, None)
