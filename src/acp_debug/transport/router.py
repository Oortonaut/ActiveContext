"""Message routing through extension chain."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from acp_debug.extension.chain import ExtensionChain
from acp_debug.transport.stdio import JsonRpcMessage
from acp_debug.types.notifications import CancelNotification, SessionUpdate
from acp_debug.types.requests import (
    CreateTerminalRequest,
    InitializeRequest,
    KillTerminalRequest,
    ListSessionsRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PermissionRequest,
    PromptRequest,
    ReadFileRequest,
    ReleaseTerminalRequest,
    SetModelRequest,
    SetModeRequest,
    TerminalOutputRequest,
    WaitForExitRequest,
    WriteFileRequest,
)
from acp_debug.types.responses import (
    CreateTerminalResponse,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PermissionResponse,
    PromptResponse,
    ReadFileResponse,
    SetModelResponse,
    SetModeResponse,
    TerminalOutputResponse,
    WaitForExitResponse,
)

# Method name to request/response type mapping
AGENT_METHODS: dict[str, tuple[type[BaseModel], type[BaseModel]]] = {
    "initialize": (InitializeRequest, InitializeResponse),
    "session/new": (NewSessionRequest, NewSessionResponse),
    "session/load": (LoadSessionRequest, LoadSessionResponse),
    "session/list": (ListSessionsRequest, ListSessionsResponse),
    "session/prompt": (PromptRequest, PromptResponse),
    "session/set_mode": (SetModeRequest, SetModeResponse),
    "session/set_model": (SetModelRequest, SetModelResponse),
}

AGENT_NOTIFICATIONS: dict[str, type[BaseModel]] = {
    "session/cancel": CancelNotification,
}

CLIENT_METHODS: dict[str, tuple[type[BaseModel], type[BaseModel] | type[None]]] = {
    "session/request_permission": (PermissionRequest, PermissionResponse),
    "fs/read_text_file": (ReadFileRequest, ReadFileResponse),
    "fs/write_text_file": (WriteFileRequest, type(None)),
    "terminal/create": (CreateTerminalRequest, CreateTerminalResponse),
    "terminal/output": (TerminalOutputRequest, TerminalOutputResponse),
    "terminal/wait_for_exit": (WaitForExitRequest, WaitForExitResponse),
    "terminal/kill": (KillTerminalRequest, type(None)),
    "terminal/release": (ReleaseTerminalRequest, type(None)),
}

CLIENT_NOTIFICATIONS: dict[str, type[BaseModel]] = {
    "session/update": SessionUpdate,
}

# Method name to chain method name mapping
METHOD_TO_CHAIN = {
    "initialize": "initialize",
    "session/new": "session_new",
    "session/load": "session_load",
    "session/list": "session_list",
    "session/prompt": "session_prompt",
    "session/set_mode": "session_set_mode",
    "session/set_model": "session_set_model",
    "session/cancel": "session_cancel",
    "session/request_permission": "request_permission",
    "fs/read_text_file": "fs_read_text_file",
    "fs/write_text_file": "fs_write_text_file",
    "terminal/create": "terminal_create",
    "terminal/output": "terminal_output",
    "terminal/wait_for_exit": "terminal_wait_for_exit",
    "terminal/kill": "terminal_kill",
    "terminal/release": "terminal_release",
    "session/update": "session_update",
}


class MessageRouter:
    """Routes JSON-RPC messages through extension chain."""

    def __init__(self, chain: ExtensionChain) -> None:
        self.chain = chain

    async def route_to_agent(
        self,
        msg: JsonRpcMessage,
        forward: Callable[[JsonRpcMessage], Awaitable[JsonRpcMessage | None]],
    ) -> JsonRpcMessage | None:
        """Route a client→agent message through extensions.

        Args:
            msg: The incoming JSON-RPC message
            forward: Function to forward the (possibly modified) message to the agent

        Returns:
            The response message, or None for notifications
        """
        method = msg.method
        if method is None:
            return None

        # Handle agent methods (requests)
        if method in AGENT_METHODS:
            req_type, resp_type = AGENT_METHODS[method]
            chain_method = METHOD_TO_CHAIN[method]

            # Parse request
            req = req_type.model_validate(msg.params or {})

            # Build final callable that forwards to agent
            async def final_forward(r: Any) -> Any:
                # Serialize back to JSON-RPC
                forward_msg = JsonRpcMessage(
                    id=msg.id,
                    method=method,
                    params=r.model_dump(by_alias=True, exclude_none=True),
                )
                response = await forward(forward_msg)
                if response and response.result is not None:
                    return resp_type.model_validate(response.result)
                return None

            # Route through chain
            chain_fn = getattr(self.chain, chain_method)
            result = await chain_fn(req, final_forward)

            # Build response
            if result is not None:
                return JsonRpcMessage(
                    id=msg.id,
                    result=result.model_dump(by_alias=True, exclude_none=True),
                )
            return JsonRpcMessage(id=msg.id, result=None)

        # Handle agent notifications
        if method in AGENT_NOTIFICATIONS:
            notif_type = AGENT_NOTIFICATIONS[method]
            chain_method = METHOD_TO_CHAIN[method]

            notif = notif_type.model_validate(msg.params or {})

            async def final_forward_notif(n: Any) -> None:
                forward_msg = JsonRpcMessage(
                    method=method,
                    params=n.model_dump(by_alias=True, exclude_none=True),
                )
                await forward(forward_msg)

            chain_fn = getattr(self.chain, chain_method)
            await chain_fn(notif, final_forward_notif)
            return None

        # Unknown method - forward as-is
        return await forward(msg)

    async def route_to_client(
        self,
        msg: JsonRpcMessage,
        forward: Callable[[JsonRpcMessage], Awaitable[JsonRpcMessage | None]],
    ) -> JsonRpcMessage | None:
        """Route an agent→client message through extensions.

        Args:
            msg: The incoming JSON-RPC message
            forward: Function to forward the (possibly modified) message to the client

        Returns:
            The response message, or None for notifications
        """
        method = msg.method
        if method is None:
            return None

        # Handle client methods (requests from agent)
        if method in CLIENT_METHODS:
            req_type, resp_type = CLIENT_METHODS[method]
            chain_method = METHOD_TO_CHAIN[method]

            req = req_type.model_validate(msg.params or {})

            async def final_forward(r: Any) -> Any:
                forward_msg = JsonRpcMessage(
                    id=msg.id,
                    method=method,
                    params=r.model_dump(by_alias=True, exclude_none=True),
                )
                response = await forward(forward_msg)
                if response and response.result is not None and hasattr(resp_type, "model_validate"):
                    return resp_type.model_validate(response.result)  # type: ignore[union-attr]
                return None

            chain_fn = getattr(self.chain, chain_method)
            result = await chain_fn(req, final_forward)

            if result is not None:
                return JsonRpcMessage(
                    id=msg.id,
                    result=result.model_dump(by_alias=True, exclude_none=True),
                )
            return JsonRpcMessage(id=msg.id, result=None)

        # Handle client notifications (from agent)
        if method in CLIENT_NOTIFICATIONS:
            notif_type = CLIENT_NOTIFICATIONS[method]
            chain_method = METHOD_TO_CHAIN[method]

            notif = notif_type.model_validate(msg.params or {})

            async def final_forward_notif(n: Any) -> None:
                forward_msg = JsonRpcMessage(
                    method=method,
                    params=n.model_dump(by_alias=True, exclude_none=True),
                )
                await forward(forward_msg)

            chain_fn = getattr(self.chain, chain_method)
            await chain_fn(notif, final_forward_notif)
            return None

        # Unknown method - forward as-is
        return await forward(msg)
