"""MCP server management menu handler.

Provides an interactive menu for managing MCP server connections
through the ConversationTransport protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from activecontext.mcp.types import MCPConnectionStatus
from activecontext.protocols.conversation import InputType

if TYPE_CHECKING:
    from activecontext.mcp.client import MCPClientManager
    from activecontext.protocols.conversation import ConversationTransport


class MCPMenuHandler:
    """Handler for /mcp slash command menu.

    Presents an interactive menu for:
    - Listing connected MCP servers
    - Listing configured (available) servers
    - Connecting to servers
    - Disconnecting from servers
    - Viewing server tools

    Example:
        >>> handler = MCPMenuHandler(mcp_manager)
        >>> result = await session.delegate_conversation(
        ...     handler,
        ...     originator="mcp:menu",
        ... )
        >>> print(f"Action: {result['action']}")
    """

    def __init__(self, mcp_manager: MCPClientManager) -> None:
        """Initialize the MCP menu handler.

        Args:
            mcp_manager: The MCP client manager from the session.
        """
        self._mcp = mcp_manager

    async def handle(self, transport: ConversationTransport) -> dict[str, Any]:
        """Handle the MCP management menu.

        Args:
            transport: Communication channel to user.

        Returns:
            Dict containing:
                - action: Last action taken ("list", "connect", "disconnect", "exit", "cancelled")
                - server: Server name if action was on a specific server
        """
        last_action = "exit"
        last_server: str | None = None

        await transport.send_output("=" * 50 + "\n")
        await transport.send_output("MCP Server Manager\n")
        await transport.send_output("=" * 50 + "\n\n")

        while True:
            # Check for cancellation
            if transport.check_cancelled():
                return {"action": "cancelled", "server": None}

            # Show current status
            await self._show_status(transport)

            # Show menu
            await transport.send_output("\nOptions:\n")
            await transport.send_output("  1. List connected servers\n")
            await transport.send_output("  2. List available servers (from config)\n")
            await transport.send_output("  3. Connect to a server\n")
            await transport.send_output("  4. Disconnect from a server\n")
            await transport.send_output("  5. View server tools\n")
            await transport.send_output("  6. Exit\n\n")

            try:
                choice = await transport.request_input(
                    "Select option (1-6): ",
                    input_type=InputType.TEXT,
                )
            except Exception:
                return {"action": "cancelled", "server": last_server}

            choice = choice.strip()

            if choice == "1":
                last_action = "list"
                await self._list_connected(transport)

            elif choice == "2":
                last_action = "list_available"
                await self._list_available(transport)

            elif choice == "3":
                server = await self._connect_server(transport)
                if server:
                    last_action = "connect"
                    last_server = server

            elif choice == "4":
                server = await self._disconnect_server(transport)
                if server:
                    last_action = "disconnect"
                    last_server = server

            elif choice == "5":
                server = await self._view_tools(transport)
                if server:
                    last_action = "view_tools"
                    last_server = server

            elif choice == "6" or choice.lower() in ("exit", "quit", "q"):
                await transport.send_output("\nExiting MCP Manager.\n")
                return {"action": last_action, "server": last_server}

            else:
                await transport.send_output(f"\nInvalid option: {choice}\n")

            await transport.send_output("\n" + "-" * 50 + "\n")

    async def _show_status(self, transport: ConversationTransport) -> None:
        """Show current connection status."""
        connections = self._mcp.list_connections()
        connected = [c for c in connections if c.status == MCPConnectionStatus.CONNECTED]

        if connected:
            await transport.send_output(f"Status: {len(connected)} server(s) connected\n")
        else:
            await transport.send_output("Status: No servers connected\n")

    async def _list_connected(self, transport: ConversationTransport) -> None:
        """List currently connected servers."""
        connections = self._mcp.list_connections()

        if not connections:
            await transport.send_output("\nNo servers connected.\n")
            return

        await transport.send_output("\nConnected Servers:\n")
        for conn in connections:
            status_icon = "✓" if conn.status == MCPConnectionStatus.CONNECTED else "○"
            tool_count = len(conn.tools) if conn.tools else 0
            await transport.send_output(
                f"  {status_icon} {conn.name} ({conn.status.value}) - {tool_count} tools\n"
            )

    async def _list_available(self, transport: ConversationTransport) -> None:
        """List available servers from config."""
        if not self._mcp.config or not self._mcp.config.servers:
            await transport.send_output("\nNo servers configured.\n")
            await transport.send_output("Add servers to your config.yaml under 'mcp.servers'.\n")
            return

        await transport.send_output("\nConfigured Servers:\n")
        for server in self._mcp.config.servers:
            # Check if connected
            conn = self._mcp.get_connection(server.name)
            if conn and conn.status == MCPConnectionStatus.CONNECTED:
                status = "✓ Connected"
            else:
                status = "○ Available"

            transport_type = server.transport
            await transport.send_output(f"  {status}: {server.name} ({transport_type})\n")

    async def _connect_server(self, transport: ConversationTransport) -> str | None:
        """Connect to a server interactively."""
        # Get available servers
        available = []
        if self._mcp.config and self._mcp.config.servers:
            for server in self._mcp.config.servers:
                conn = self._mcp.get_connection(server.name)
                if not conn or conn.status != MCPConnectionStatus.CONNECTED:
                    available.append(server.name)

        if not available:
            await transport.send_output(
                "\nNo available servers to connect. All configured servers are connected.\n"
            )
            return None

        await transport.send_output("\nAvailable servers:\n")
        for i, name in enumerate(available, 1):
            await transport.send_output(f"  {i}. {name}\n")

        try:
            choice = await transport.request_input(
                "Enter server number or name (or 'cancel'): ",
                input_type=InputType.TEXT,
            )
        except Exception:
            return None

        choice = choice.strip()
        if choice.lower() == "cancel":
            return None

        # Resolve server name
        server_name: str | None = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                server_name = available[idx]
        elif choice in available:
            server_name = choice

        if not server_name:
            await transport.send_output(f"Invalid selection: {choice}\n")
            return None

        # Connect
        await transport.send_output(f"\nConnecting to {server_name}...")
        await transport.send_progress(0, 1, status="Connecting")

        try:
            conn = await self._mcp.connect(name=server_name)
            await transport.send_progress(1, 1, status="Connected")
            await transport.send_output(f" Connected! ({len(conn.tools)} tools available)\n")
            return server_name
        except Exception as e:
            await transport.send_output(f" Failed: {e}\n")
            return None

    async def _disconnect_server(self, transport: ConversationTransport) -> str | None:
        """Disconnect from a server interactively."""
        connections = self._mcp.list_connections()
        connected = [c.name for c in connections if c.status == MCPConnectionStatus.CONNECTED]

        if not connected:
            await transport.send_output("\nNo servers to disconnect.\n")
            return None

        await transport.send_output("\nConnected servers:\n")
        for i, name in enumerate(connected, 1):
            await transport.send_output(f"  {i}. {name}\n")

        try:
            choice = await transport.request_input(
                "Enter server number or name (or 'cancel'): ",
                input_type=InputType.TEXT,
            )
        except Exception:
            return None

        choice = choice.strip()
        if choice.lower() == "cancel":
            return None

        # Resolve server name
        server_name: str | None = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(connected):
                server_name = connected[idx]
        elif choice in connected:
            server_name = choice

        if not server_name:
            await transport.send_output(f"Invalid selection: {choice}\n")
            return None

        # Disconnect
        await transport.send_output(f"\nDisconnecting from {server_name}...")

        try:
            await self._mcp.disconnect(server_name)
            await transport.send_output(" Disconnected.\n")
            return server_name
        except Exception as e:
            await transport.send_output(f" Failed: {e}\n")
            return None

    async def _view_tools(self, transport: ConversationTransport) -> str | None:
        """View tools from a connected server."""
        connections = self._mcp.list_connections()
        connected = [c for c in connections if c.status == MCPConnectionStatus.CONNECTED]

        if not connected:
            await transport.send_output("\nNo servers connected.\n")
            return None

        await transport.send_output("\nConnected servers:\n")
        for i, conn in enumerate(connected, 1):
            await transport.send_output(f"  {i}. {conn.name} ({len(conn.tools)} tools)\n")

        try:
            choice = await transport.request_input(
                "Enter server number or name (or 'cancel'): ",
                input_type=InputType.TEXT,
            )
        except Exception:
            return None

        choice = choice.strip()
        if choice.lower() == "cancel":
            return None

        # Resolve server
        target_conn = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(connected):
                target_conn = connected[idx]
        else:
            for conn in connected:
                if conn.name == choice:
                    target_conn = conn
                    break

        if not target_conn:
            await transport.send_output(f"Invalid selection: {choice}\n")
            return None

        # Display tools
        await transport.send_output(f"\nTools from {target_conn.name}:\n")
        await transport.send_output("-" * 40 + "\n")

        if not target_conn.tools:
            await transport.send_output("  (no tools)\n")
        else:
            for tool in target_conn.tools:
                await transport.send_output(f"\n  {tool.name}\n")
                if tool.description:
                    # Wrap description
                    desc_lines = tool.description.split("\n")
                    for line in desc_lines[:3]:  # First 3 lines
                        await transport.send_output(f"    {line}\n")
                    if len(desc_lines) > 3:
                        await transport.send_output("    ...\n")

        return target_conn.name
