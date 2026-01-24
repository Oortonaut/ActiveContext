"""MCP permission management."""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger("activecontext.mcp.permissions")


@dataclass
class MCPPermissionDenied(Exception):
    """Raised when an MCP tool call is denied."""

    server_name: str
    tool_name: str
    arguments: dict[str, Any]

    def __str__(self) -> str:
        return f"MCP tool call denied: {self.server_name}.{self.tool_name}"


@dataclass
class MCPPermissionRule:
    """A permission rule for MCP tool access.

    Patterns support glob syntax:
        - "filesystem.*" matches all tools on filesystem server
        - "*.read_*" matches read tools on any server
        - "github.search_repos" matches specific tool
    """

    pattern: str
    allow: bool = True


@dataclass
class MCPPermissionManager:
    """Manages permissions for MCP tool calls.

    Supports:
        - Glob-pattern rules for server.tool matching
        - Temporary grants for "allow once" flow
        - Default deny policy (configurable)
    """

    rules: list[MCPPermissionRule] = field(default_factory=list)
    deny_by_default: bool = True
    _temporary_grants: set[str] = field(default_factory=set)

    def check_access(self, server_name: str, tool_name: str) -> bool:
        """Check if a tool call is permitted.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call

        Returns:
            True if access is allowed, False otherwise
        """
        full_name = f"{server_name}.{tool_name}"

        # Check temporary grants first
        if full_name in self._temporary_grants:
            return True
        if f"{server_name}.*" in self._temporary_grants:
            return True
        if "*.*" in self._temporary_grants:
            return True

        # Check rules (first match wins)
        for rule in self.rules:
            if fnmatch.fnmatch(full_name, rule.pattern):
                return rule.allow

        return not self.deny_by_default

    def grant_temporary(self, server_name: str, tool_name: str | None = None) -> None:
        """Grant temporary access for a tool or all tools on a server.

        Args:
            server_name: Server name to grant access to
            tool_name: Optional specific tool name. If None, grants access
                       to all tools on the server.
        """
        if tool_name:
            self._temporary_grants.add(f"{server_name}.{tool_name}")
            _log.debug(f"Granted temporary access to {server_name}.{tool_name}")
        else:
            self._temporary_grants.add(f"{server_name}.*")
            _log.debug(f"Granted temporary access to all {server_name} tools")

    def revoke_temporary(self, server_name: str, tool_name: str | None = None) -> None:
        """Revoke temporary access."""
        if tool_name:
            self._temporary_grants.discard(f"{server_name}.{tool_name}")
        else:
            self._temporary_grants.discard(f"{server_name}.*")

    def clear_temporary_grants(self) -> None:
        """Clear all temporary grants."""
        self._temporary_grants.clear()
        _log.debug("Cleared all temporary MCP grants")

    def add_rule(self, pattern: str, allow: bool = True) -> None:
        """Add a permission rule.

        Rules are evaluated in order, first match wins.

        Args:
            pattern: Glob pattern for server.tool (e.g., "filesystem.*")
            allow: Whether to allow or deny matching calls
        """
        self.rules.append(MCPPermissionRule(pattern=pattern, allow=allow))

    def allow_server(self, server_name: str) -> None:
        """Add a rule to allow all tools on a server."""
        self.add_rule(f"{server_name}.*", allow=True)

    def deny_server(self, server_name: str) -> None:
        """Add a rule to deny all tools on a server."""
        self.add_rule(f"{server_name}.*", allow=False)
