"""Mock agent mode - replace agent with mock responses."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

from acp_debug.extension.chain import ExtensionChain
from acp_debug.extension.loader import load_extensions
from acp_debug.extension.mock_agent import MockAgentBase
from acp_debug.transport.stdio import JsonRpcMessage, StdioTransport

if TYPE_CHECKING:
    from acp_debug.config import Config

console = Console(stderr=True)


async def run_mock_agent(config: Config, script: Path | None) -> int:
    """Run in mock agent mode.

    Args:
        config: Configuration
        script: Optional script defining mock responses

    Returns:
        Exit code
    """
    # Load extensions
    extensions = load_extensions(config)

    # Find mock agent extension or use default
    mock_agent: MockAgentBase | None = None
    for ext in extensions:
        if isinstance(ext, MockAgentBase):
            mock_agent = ext
            break

    if mock_agent is None:
        mock_agent = MockAgentBase()
        extensions.insert(0, mock_agent)

    if not config.quiet:
        console.print(f"[dim]Mock agent: {mock_agent.name} v{mock_agent.version}[/dim]")
        console.print(f"[dim]Loaded {len(extensions)} extension(s)[/dim]")

    # Create extension chain
    chain = ExtensionChain(extensions)
    chain.initialize_all()

    # Create transport for IDE communication
    transport = await StdioTransport.from_stdio()

    if not config.quiet:
        console.print("[green]Mock agent running[/green]")

    try:
        async for msg in transport.messages():
            response = await _handle_message(chain, mock_agent, msg)
            if response:
                await transport.write_message(response)
    except KeyboardInterrupt:
        pass
    finally:
        chain.shutdown_all()
        await transport.close()

    return 0


async def _handle_message(
    chain: ExtensionChain,
    mock: MockAgentBase,
    msg: JsonRpcMessage,
) -> JsonRpcMessage | None:
    """Handle an incoming message using the mock agent."""
    from acp_debug.transport.router import AGENT_METHODS, AGENT_NOTIFICATIONS, METHOD_TO_CHAIN

    method = msg.method
    if method is None:
        return None

    # Handle requests
    if method in AGENT_METHODS:
        req_type, resp_type = AGENT_METHODS[method]
        chain_method = METHOD_TO_CHAIN[method]

        req = req_type.model_validate(msg.params or {})

        # For mock agent, the "final" just calls the mock's method directly
        mock_method = getattr(mock, chain_method)

        async def final(r: Any) -> Any:
            # Mock agent handles it directly
            return await mock_method(r, _noop)

        chain_fn = getattr(chain, chain_method)
        result = await chain_fn(req, final)

        if result is not None:
            return JsonRpcMessage(
                id=msg.id,
                result=result.model_dump(by_alias=True, exclude_none=True),
            )
        return JsonRpcMessage(id=msg.id, result=None)

    # Handle notifications
    if method in AGENT_NOTIFICATIONS:
        notif_type = AGENT_NOTIFICATIONS[method]
        chain_method = METHOD_TO_CHAIN[method]

        notif = notif_type.model_validate(msg.params or {})

        async def final_notif(n: Any) -> None:
            pass  # Notifications don't need forwarding in mock mode

        chain_fn = getattr(chain, chain_method)
        await chain_fn(notif, final_notif)
        return None

    # Unknown method
    return JsonRpcMessage(
        id=msg.id,
        error={"code": -32601, "message": f"Method not found: {method}"},
    )


async def _noop(x: Any) -> Any:
    """No-op callable for mock methods."""
    return x
