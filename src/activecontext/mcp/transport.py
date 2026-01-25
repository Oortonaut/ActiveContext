"""MCP transport factory for stdio and streamable-http transports."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, AsyncContextManager

if TYPE_CHECKING:
    from activecontext.config.schema import MCPServerConfig


def _expand_env_vars(env: dict[str, str]) -> dict[str, str]:
    """Expand environment variable references in env dict values.

    Supports ${VAR} syntax for referencing environment variables.
    """
    result = {}
    for key, value in env.items():
        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            result[key] = os.environ.get(var_name, "")
        else:
            result[key] = value
    return result


async def create_transport(
    config: MCPServerConfig,
) -> AsyncContextManager[tuple[Any, Any, Any]]:
    """Create appropriate transport based on config.

    Args:
        config: MCP server configuration

    Returns:
        Async context manager yielding (read_stream, write_stream, session_info)

    Raises:
        ValueError: If transport type is invalid or required fields are missing
    """
    if config.transport == "stdio":
        if not config.command:
            raise ValueError(f"stdio transport requires 'command' for server '{config.name}'")

        from mcp.client.stdio import StdioServerParameters, stdio_client

        # Merge base environment with config env (config takes precedence)
        merged_env = dict(os.environ)
        merged_env.update(_expand_env_vars(config.env))

        params = StdioServerParameters(
            command=config.command[0],
            args=config.command[1:] + config.extra_args,
            env=merged_env,
        )
        return stdio_client(params)

    elif config.transport == "streamable-http":
        if not config.url:
            raise ValueError(f"streamable-http transport requires 'url' for server '{config.name}'")

        from mcp.client.streamable_http import streamablehttp_client

        return streamablehttp_client(config.url)

    elif config.transport == "sse":
        if not config.url:
            raise ValueError(f"sse transport requires 'url' for server '{config.name}'")

        from mcp.client.sse import sse_client

        # Expand env vars in headers and filter None values
        headers = {k: v for k, v in _expand_env_vars(config.headers).items() if v}
        return sse_client(
            config.url,
            headers=headers if headers else None,
            timeout=config.timeout,
        )

    else:
        raise ValueError(f"Unknown transport: {config.transport}")
