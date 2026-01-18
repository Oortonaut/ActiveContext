"""ACP Agent implementation for ActiveContext.

This module provides the ACP-compatible agent that can be used with
Rider, Zed, and other ACP-supporting editors.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import acp
from acp import helpers
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    Implementation,
    ModelInfo,
    PermissionOption,
    PromptCapabilities,
    SessionCapabilities,
    SessionMode,
    SessionModelState,
    SessionModeState,
    SetSessionModelResponse,
    SetSessionModeResponse,
    TextContentBlock,
    ToolCallUpdate,
)

from activecontext.core.llm import (
    LiteLLMProvider,
    get_available_models,
    get_default_model,
)
from activecontext.logging import get_logger
from activecontext.session.protocols import UpdateKind
from activecontext.session.session_manager import Session, SessionManager
from activecontext.session.storage import list_sessions as list_sessions_from_disk
from activecontext.terminal.acp_executor import ACPTerminalExecutor

log = get_logger("acp")

if TYPE_CHECKING:
    from acp.interfaces import Client

# Default session modes (used if no config or config has no modes)
DEFAULT_SESSION_MODES = [
    SessionMode(id="normal", name="Normal", description="Standard mode"),
    SessionMode(id="plan", name="Plan", description="Plan before acting"),
    SessionMode(id="brave", name="Brave", description="Autonomous with fewer confirmations"),
]
DEFAULT_MODE_ID = "normal"


class ActiveContextAgent:
    """ACP Agent adapter for ActiveContext.

    This implements the ACP Agent protocol by delegating to
    SessionManager for actual session management.
    """

    def __init__(self) -> None:
        # Load config for session modes
        self._session_modes = DEFAULT_SESSION_MODES
        self._default_mode_id = DEFAULT_MODE_ID
        self._load_modes_from_config()

        # Set up LLM provider based on available API keys
        default_model = get_default_model()
        default_llm: LiteLLMProvider | None = None
        if default_model:
            default_llm = LiteLLMProvider(default_model)

        # Note: type ignore due to Protocol/async-generator limitation in mypy
        self._manager = SessionManager(default_llm=default_llm)  # type: ignore[arg-type]
        self._conn: Client | None = None
        self._sessions_cwd: dict[str, str] = {}  # session_id -> cwd
        self._sessions_model: dict[str, str] = {}  # session_id -> model_id
        self._sessions_mode: dict[str, str] = {}  # session_id -> mode_id
        self._current_model_id = default_model
        self._current_cwd: str | None = None  # Track most recent cwd for list_sessions

        # Nagle-style batching for RESPONSE_CHUNK updates
        self._chunk_buffers: dict[str, str] = {}  # session_id -> accumulated text
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}  # session_id -> flush task
        self._batch_enabled = True  # Can be disabled via config
        self._flush_interval = 0.05  # 50ms
        self._flush_threshold = 100  # characters
        self._load_batch_config()

    def _load_modes_from_config(self) -> None:
        """Load session modes from config if available."""
        try:
            from activecontext.config import get_config

            config = get_config()
            if config.session.modes:
                self._session_modes = [
                    SessionMode(
                        id=m.id,
                        name=m.name,
                        description=m.description,
                    )
                    for m in config.session.modes
                ]
                # Use config default or first mode
                if config.session.default_mode:
                    self._default_mode_id = config.session.default_mode
                elif self._session_modes:
                    self._default_mode_id = self._session_modes[0].id
        except ImportError:
            pass  # Config module not available
        except Exception:
            pass  # Config loading failed, use defaults

    def _load_batch_config(self) -> None:
        """Load batching settings from config if available."""
        try:
            from activecontext.config import get_config

            config = get_config()
            acp_config = config.extra.get("acp", {})
            batch_config = acp_config.get("batching", {})

            if "enabled" in batch_config:
                self._batch_enabled = bool(batch_config["enabled"])
            if "flush_interval" in batch_config:
                self._flush_interval = float(batch_config["flush_interval"])
            if "flush_threshold" in batch_config:
                self._flush_threshold = int(batch_config["flush_threshold"])
        except ImportError:
            pass
        except Exception:
            pass

    # --- Nagle-style batching for RESPONSE_CHUNK ---

    async def _flush_chunks(self, session_id: str) -> None:
        """Flush accumulated response chunks for a session."""
        text = self._chunk_buffers.pop(session_id, "")
        self._flush_tasks.pop(session_id, None)

        if text and self._conn:
            await self._conn.session_update(
                session_id,
                acp.update_agent_message_text(text),
            )

    async def _buffer_chunk(self, session_id: str, text: str) -> None:
        """Buffer a response chunk, flushing if threshold reached."""
        # Append to buffer
        self._chunk_buffers[session_id] = self._chunk_buffers.get(session_id, "") + text

        # Check size threshold - flush immediately if exceeded
        if len(self._chunk_buffers[session_id]) >= self._flush_threshold:
            if session_id in self._flush_tasks:
                self._flush_tasks[session_id].cancel()
            await self._flush_chunks(session_id)
            return

        # Schedule flush if not already scheduled
        if session_id not in self._flush_tasks:
            self._flush_tasks[session_id] = asyncio.create_task(
                self._delayed_flush(session_id)
            )

    async def _delayed_flush(self, session_id: str) -> None:
        """Flush after delay (Nagle timer)."""
        await asyncio.sleep(self._flush_interval)
        await self._flush_chunks(session_id)

    def _cleanup_session_buffers(self, session_id: str) -> None:
        """Clean up batching state for a closed session."""
        self._chunk_buffers.pop(session_id, None)
        task = self._flush_tasks.pop(session_id, None)
        if task:
            task.cancel()

    # --- End batching ---

    # --- File permission requests ---

    async def _request_file_permission(
        self, session_id: str, path: str, mode: str
    ) -> tuple[bool, bool]:
        """Request file access permission from user via ACP.

        This is called when the Timeline catches a PermissionDenied exception.
        The user is prompted to allow or deny access.

        Args:
            session_id: The session requesting permission.
            path: Absolute path to the file.
            mode: "read" or "write".

        Returns:
            (granted, persist) - granted=True if allowed, persist=True if "allow_always"
        """
        if not self._conn:
            log.warning("No ACP client connected, denying permission request")
            return (False, False)

        tool_call = ToolCallUpdate(
            tool_call_id=f"file-access-{uuid.uuid4()}",
            title=f"File {mode} access",
            kind="read" if mode == "read" else "edit",
            status="pending",
            content=[
                helpers.tool_content(
                    helpers.text_block(
                        f"The agent wants to {mode} the file:\n{path}"
                    )
                )
            ],
        )

        options = [
            PermissionOption(option_id="allow_once", name="Allow once", kind="allow_once"),
            PermissionOption(
                option_id="allow_always", name="Allow always", kind="allow_always"
            ),
            PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
        ]

        try:
            response = await self._conn.request_permission(
                options=options,
                session_id=session_id,
                tool_call=tool_call,
            )

            if response.outcome.outcome == "selected":
                option_id = response.outcome.option_id
                if option_id == "allow_once":
                    log.info("Permission granted (once) for %s: %s", mode, path)
                    return (True, False)
                elif option_id == "allow_always":
                    log.info("Permission granted (always) for %s: %s", mode, path)
                    return (True, True)

            log.info("Permission denied for %s: %s", mode, path)
            return (False, False)

        except Exception as e:
            log.error("Permission request failed: %s", e)
            return (False, False)

    # --- End file permission requests ---

    # --- Shell permission requests ---

    async def _request_shell_permission(
        self, session_id: str, command: str, args: list[str] | None
    ) -> tuple[bool, bool]:
        """Request shell command permission from user via ACP.

        This is called when the Timeline catches a shell permission denied.
        The user is prompted to allow or deny the command.

        Args:
            session_id: The session requesting permission.
            command: The command to execute.
            args: Optional command arguments.

        Returns:
            (granted, persist) - granted=True if allowed, persist=True if "allow_always"
        """
        if not self._conn:
            log.warning("No ACP client connected, denying shell permission request")
            return (False, False)

        full_command = f"{command} {' '.join(args or [])}" if args else command

        tool_call = ToolCallUpdate(
            tool_call_id=f"shell-access-{uuid.uuid4()}",
            title="Shell command",
            kind="execute",  # "execute" is a valid ToolCallKind for shell commands
            status="pending",
            content=[
                helpers.tool_content(
                    helpers.text_block(
                        f"The agent wants to run:\n```\n{full_command}\n```"
                    )
                )
            ],
        )

        options = [
            PermissionOption(option_id="allow_once", name="Allow once", kind="allow_once"),
            PermissionOption(
                option_id="allow_always", name="Allow always", kind="allow_always"
            ),
            PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
        ]

        try:
            response = await self._conn.request_permission(
                options=options,
                session_id=session_id,
                tool_call=tool_call,
            )

            if response.outcome.outcome == "selected":
                option_id = response.outcome.option_id
                if option_id == "allow_once":
                    log.info("Shell permission granted (once): %s", full_command)
                    return (True, False)
                elif option_id == "allow_always":
                    log.info("Shell permission granted (always): %s", full_command)
                    return (True, True)

            log.info("Shell permission denied: %s", full_command)
            return (False, False)

        except Exception as e:
            log.error("Shell permission request failed: %s", e)
            return (False, False)


    # --- End shell permission requests ---

    # --- Website permission requests ---

    async def _request_website_permission(
        self, session_id: str, url: str, method: str
    ) -> tuple[bool, bool]:
        """Request website access permission from user via ACP.

        This is called when the Timeline catches a website permission denied.
        The user is prompted to allow or deny the request.

        Args:
            session_id: The session requesting permission.
            url: The URL to access.
            method: HTTP method (GET, POST, etc.).

        Returns:
            (granted, persist) - granted=True if allowed, persist=True if "allow_always"
        """
        if not self._conn:
            log.warning("No ACP client connected, denying website permission request")
            return (False, False)

        from urllib.parse import urlparse

        parsed = urlparse(url)

        tool_call = ToolCallUpdate(
            tool_call_id=f"website-access-{uuid.uuid4()}",
            title=f"Website {method} request",
            kind="read" if method == "GET" else "edit",
            status="pending",
            content=[
                helpers.tool_content(
                    helpers.text_block(
                        f"The agent wants to make a {method} request to:\n{url}\nDomain: {parsed.netloc}"
                    )
                )
            ],
        )

        options = [
            PermissionOption(option_id="allow_once", name="Allow once", kind="allow_once"),
            PermissionOption(
                option_id="allow_always", name="Allow always", kind="allow_always"
            ),
            PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
        ]

        try:
            response = await self._conn.request_permission(
                options=options,
                session_id=session_id,
                tool_call=tool_call,
            )

            if response.outcome.outcome == "selected":
                option_id = response.outcome.option_id
                if option_id == "allow_once":
                    log.info("Website permission granted (once): %s %s", method, url)
                    return (True, False)
                elif option_id == "allow_always":
                    log.info("Website permission granted (always): %s %s", method, url)
                    return (True, True)

            log.info("Website permission denied: %s %s", method, url)
            return (False, False)

        except Exception as e:
            log.error("Website permission request failed: %s", e)
            return (False, False)

    # --- End website permission requests ---

    # --- Import permission requests ---

    async def _request_import_permission(
        self, session_id: str, module: str
    ) -> tuple[bool, bool, bool]:
        """Request import permission from user via ACP.

        This is called when the Timeline catches an ImportDenied exception.
        The user is prompted to allow or deny the import.

        Args:
            session_id: The session requesting permission.
            module: The module name that was denied.

        Returns:
            (granted, persist, include_submodules) -
            granted=True if allowed,
            persist=True if "allow_always",
            include_submodules=True if "allow with submodules"
        """
        if not self._conn:
            log.warning("No ACP client connected, denying import permission request")
            return (False, False, False)

        # Get the top-level module for display
        top_level = module.split(".")[0]
        display_module = module if module == top_level else f"{module} (top-level: {top_level})"

        tool_call = ToolCallUpdate(
            tool_call_id=f"import-access-{uuid.uuid4()}",
            title=f"Import '{top_level}'",
            kind="execute",
            status="pending",
            content=[
                helpers.tool_content(
                    helpers.text_block(
                        f"The agent wants to import:\n```\nimport {display_module}\n```"
                    )
                )
            ],
        )

        options = [
            PermissionOption(option_id="allow_once", name="Allow once", kind="allow_once"),
            PermissionOption(
                option_id="allow_always", name="Allow always (this module only)", kind="allow_always"
            ),
            PermissionOption(
                option_id="allow_always_submodules", name="Allow always (+ submodules)", kind="allow_always"
            ),
            PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
        ]

        try:
            response = await self._conn.request_permission(
                options=options,
                session_id=session_id,
                tool_call=tool_call,
            )

            if response.outcome.outcome == "selected":
                option_id = response.outcome.option_id
                if option_id == "allow_once":
                    log.info("Import permission granted (once): %s", module)
                    return (True, False, False)
                elif option_id == "allow_always":
                    log.info("Import permission granted (always, module only): %s", module)
                    return (True, True, False)
                elif option_id == "allow_always_submodules":
                    log.info("Import permission granted (always, + submodules): %s", module)
                    return (True, True, True)

            log.info("Import permission denied: %s", module)
            return (False, False, False)

        except Exception as e:
            log.error("Import permission request failed: %s", e)
            return (False, False, False)

    # --- End import permission requests ---

    def on_connect(self, conn: Client) -> None:
        """Called when a client connects."""
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> acp.InitializeResponse:
        """Handle initialization request from client."""
        return acp.InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(
                name="activecontext",
                version="0.1.0",
            ),
            agent_capabilities=AgentCapabilities(
                load_session=True,  # Session persistence enabled
                prompt_capabilities=PromptCapabilities(
                    image=False,
                    audio=False,
                ),
                session_capabilities=SessionCapabilities(),
            ),
        )

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp.NewSessionResponse:
        """Create a new session."""
        import traceback

        try:
            # Track current cwd for list_sessions
            self._current_cwd = cwd

            # Create permission requester callbacks bound to this agent
            permission_requester = self._request_file_permission
            shell_permission_requester = self._request_shell_permission
            website_permission_requester = self._request_website_permission
            import_permission_requester = self._request_import_permission

            session = await self._manager.create_session(
                cwd=cwd,
                permission_requester=permission_requester,
                shell_permission_requester=shell_permission_requester,
                website_permission_requester=website_permission_requester,
                import_permission_requester=import_permission_requester,
            )

            # Now create ACP terminal executor with the actual session_id
            if self._conn:
                terminal_executor = ACPTerminalExecutor(
                    client=self._conn,
                    session_id=session.session_id,
                    default_cwd=cwd,
                )
                # Update the timeline with the ACP executor
                session.timeline._terminal_executor = terminal_executor

            self._sessions_cwd[session.session_id] = cwd
            self._sessions_mode[session.session_id] = self._default_mode_id

            # Save session to disk immediately
            try:
                session.save()
                log.debug("Saved new session %s to disk", session.session_id)
            except Exception as e:
                log.warning("Failed to save session to disk: %s", e)

            # Build available models from environment
            available = get_available_models()
            models_state = None
            if available:
                current_model = self._current_model_id or available[0].model_id
                self._sessions_model[session.session_id] = current_model
                models_state = SessionModelState(
                    available_models=[
                        ModelInfo(
                            model_id=m.model_id,
                            name=m.name,
                            description=m.description,
                        )
                        for m in available
                    ],
                    current_model_id=current_model,
                )

            # Build session modes
            modes_state = SessionModeState(
                available_modes=self._session_modes,
                current_mode_id=self._default_mode_id,
            )

            model_ids = [m.model_id for m in models_state.available_models] if models_state else []
            mode_ids = [m.id for m in modes_state.available_modes]
            
            log.info("Created session %s", session.session_id)
            log.info("  Models (%d): %s", len(model_ids), ", ".join(model_ids) if model_ids else "none")
            log.info("  Modes (%d): %s", len(mode_ids), ", ".join(mode_ids))
            
            return acp.NewSessionResponse(
                session_id=session.session_id,
                models=models_state,
                modes=modes_state,
            )

        except Exception as e:
            # Log the full error with traceback
            log.error("Failed to create session: %s", e)
            log.error("%s", traceback.format_exc())
            
            # Return proper ACP error to the IDE instead of hanging
            raise acp.RequestError(
                code=-32603,  # Internal error
                message="Failed to initialize ACP session",
                data={"details": str(e)},
            )

    async def load_session(
        self,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        session_id: str = "",
        **kwargs: Any,
    ) -> acp.LoadSessionResponse | None:
        """Load a previously persisted session."""
        import traceback

        try:
            # Track current cwd
            self._current_cwd = cwd

            # Check if already loaded in memory
            existing = await self._manager.get_session(session_id)
            if existing:
                log.info("Session %s already loaded", session_id)
                return acp.LoadSessionResponse(
                    session_id=session_id,
                    modes=SessionModeState(
                        available_modes=self._session_modes,
                        current_mode_id=self._sessions_mode.get(session_id, self._default_mode_id),
                    ),
                )

            # Load from disk
            session = Session.from_file(
                cwd=cwd,
                session_id=session_id,
                llm=self._manager._default_llm,
            )

            if not session:
                log.warning("Session %s not found on disk", session_id)
                return None

            # Register with manager
            self._manager._sessions[session_id] = session

            # Set up ACP terminal executor
            if self._conn:
                terminal_executor = ACPTerminalExecutor(
                    client=self._conn,
                    session_id=session_id,
                    default_cwd=cwd,
                )
                session.timeline._terminal_executor = terminal_executor

            # Track session metadata
            self._sessions_cwd[session_id] = cwd
            self._sessions_mode[session_id] = self._default_mode_id

            # Build model state
            available = get_available_models()
            models_state = None
            if available:
                current_model = self._current_model_id or available[0].model_id
                self._sessions_model[session_id] = current_model
                models_state = SessionModelState(
                    available_models=[
                        ModelInfo(
                            model_id=m.model_id,
                            name=m.name,
                            description=m.description,
                        )
                        for m in available
                    ],
                    current_model_id=current_model,
                )

            log.info("Loaded session %s from disk (title: %s)", session_id, session.title)

            return acp.LoadSessionResponse(
                session_id=session_id,
                models=models_state,
                modes=SessionModeState(
                    available_modes=self._session_modes,
                    current_mode_id=self._default_mode_id,
                ),
            )

        except Exception as e:
            log.error("Failed to load session %s: %s", session_id, e)
            log.error("%s", traceback.format_exc())
            return None

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> acp.schema.ListSessionsResponse:
        """List all sessions (both active and persisted on disk)."""
        # Determine which cwd to use
        target_cwd = cwd or self._current_cwd
        if not target_cwd:
            # No cwd available, return empty
            return acp.schema.ListSessionsResponse(sessions=[])

        # Get persisted sessions from disk
        persisted = list_sessions_from_disk(target_cwd)

        # Build session info list
        sessions = []
        for meta in persisted:
            sessions.append(
                acp.schema.SessionInfo(
                    session_id=meta.session_id,
                    cwd=meta.cwd,
                    # Include title and timestamps in custom fields if ACP supports it
                )
            )

        return acp.schema.ListSessionsResponse(sessions=sessions)

    async def set_session_mode(
        self,
        mode_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse | None:
        """Set session mode."""
        # Validate mode exists
        valid_modes = {m.id for m in self._session_modes}
        if mode_id not in valid_modes:
            raise acp.RequestError(
                code=-32602,
                message=f"Invalid mode: {mode_id}. Valid modes: {valid_modes}",
            )

        self._sessions_mode[session_id] = mode_id
        return SetSessionModeResponse()

    async def set_session_model(
        self,
        model_id: str,
        session_id: str,
        **kwargs: Any,
    ) -> SetSessionModelResponse | None:
        """Set session model - switches the LLM provider for this session."""
        session = await self._manager.get_session(session_id)
        if not session:
            raise acp.RequestError(
                code=-32600,
                message=f"Session not found: {session_id}",
            )

        # Create new LLM provider with the requested model
        new_llm = LiteLLMProvider(model_id)
        session.set_llm(new_llm)  # type: ignore[arg-type]
        self._sessions_model[session_id] = model_id

        return SetSessionModelResponse()

    async def authenticate(
        self,
        method_id: str,
        **kwargs: Any,
    ) -> acp.AuthenticateResponse | None:
        """Handle authentication (not implemented)."""
        return None

    async def prompt(
        self,
        prompt: list[Any],
        session_id: str,
        **kwargs: Any,
    ) -> acp.PromptResponse:
        """Handle a prompt request.

        This is the main interaction method. It:
        1. Gets the session
        2. Processes the prompt through the session
        3. Streams updates back via session_update
        4. Returns the final response
        """
        import traceback

        session = await self._manager.get_session(session_id)
        if not session:
            raise acp.RequestError(
                code=-32600,
                message=f"Session not found: {session_id}",
            )

        # Extract text from prompt blocks
        content = ""
        for block in prompt:
            if isinstance(block, TextContentBlock) or hasattr(block, "text"):
                content += block.text

        # Handle slash commands before LLM sees them
        if content.strip().startswith("/"):
            handled, response = await self._handle_slash_command(
                content.strip(), session_id
            )
            if handled:
                if self._conn and response:
                    await self._conn.session_update(
                        session_id,
                        acp.update_agent_message_text(response),
                    )
                return acp.PromptResponse(stop_reason="end_turn")

        # Process prompt and stream updates
        try:
            response_text = ""
            async for update in session.prompt(content):
                if self._conn:
                    await self._emit_update(session_id, update)

                # Collect response chunks
                if update.kind == UpdateKind.RESPONSE_CHUNK:
                    response_text += update.payload.get("text", "")
                elif update.kind == UpdateKind.STATEMENT_EXECUTED:
                    # Include execution output in response
                    stdout = update.payload.get("stdout", "")
                    if stdout:
                        response_text += f"\n{stdout}"
        except Exception as e:
            # Log the error but don't crash the agent
            log.error("Error in prompt: %s", e)
            log.error("%s", traceback.format_exc())
            # Send error message to user
            if self._conn:
                await self._conn.session_update(
                    session_id,
                    acp.update_agent_message_text(f"\n\nError: {e}"),
                )

        # Auto-save session after each prompt
        try:
            session.save()
            log.debug("Auto-saved session %s", session_id)
        except Exception as e:
            log.warning("Failed to auto-save session: %s", e)

        return acp.PromptResponse(stop_reason="end_turn")

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp.schema.ForkSessionResponse:
        """Fork a session (not implemented)."""
        raise acp.RequestError(
            code=-32601,
            message="Fork session not implemented",
        )

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp.schema.ResumeSessionResponse:
        """Resume a session."""
        session = await self._manager.get_session(session_id)
        if not session:
            raise acp.RequestError(
                code=-32600,
                message=f"Session not found: {session_id}",
            )
        return acp.schema.ResumeSessionResponse()

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Cancel the current operation in a session."""
        # Flush any pending chunks before canceling
        if session_id in self._chunk_buffers:
            if session_id in self._flush_tasks:
                self._flush_tasks[session_id].cancel()
            await self._flush_chunks(session_id)

        session = await self._manager.get_session(session_id)
        if session:
            await session.cancel()

    async def ext_method(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle extension methods."""
        return {}

    async def ext_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Handle extension notifications."""
        pass

    async def _handle_slash_command(
        self, content: str, session_id: str
    ) -> tuple[bool, str]:
        """Handle slash commands before they reach the LLM.

        Returns:
            (handled, response) - handled=True if command was processed
        """
        import os

        parts = content.split(maxsplit=1)
        command = parts[0].lower()
        # args = parts[1] if len(parts) > 1 else ""

        if command == "/exit":
            log.info("/exit command received, shutting down")
            # Flush any pending chunks before closing
            if session_id in self._chunk_buffers:
                if session_id in self._flush_tasks:
                    self._flush_tasks[session_id].cancel()
                await self._flush_chunks(session_id)
            # Clean shutdown
            await self._manager.close_session(session_id)
            # Give time for response to be sent, then exit
            asyncio.get_event_loop().call_later(0.5, lambda: os._exit(0))
            return True, "Goodbye!"

        elif command == "/help":
            return True, (
                "Available commands:\n"
                "  /exit      - Shutdown the agent\n"
                "  /help      - Show this help\n"
                "  /clear     - Clear conversation history\n"
                "  /context   - Show current context objects\n"
                "  /title     - Set session title\n"
                "  /dashboard - Open monitoring dashboard (start/stop/status/open)"
            )

        elif command == "/title":
            args = parts[1] if len(parts) > 1 else ""
            if not args.strip():
                # Show current title
                session = await self._manager.get_session(session_id)
                if session:
                    return True, f"Current title: {session.title}"
                return True, "Session not found."

            # Set new title
            session = await self._manager.get_session(session_id)
            if session:
                session.set_title(args.strip())
                # Save immediately to persist the title
                try:
                    session.save()
                except Exception as e:
                    log.warning("Failed to save session after title change: %s", e)
                return True, f"Session title set to: {args.strip()}"
            return True, "Session not found."

        elif command == "/clear":
            session = await self._manager.get_session(session_id)
            if session:
                session.clear_conversation()
            return True, "Conversation history cleared."

        elif command == "/context":
            session = await self._manager.get_session(session_id)
            if session:
                objects = session.get_context_objects()
                if objects:
                    lines = ["Current context objects:"]
                    for obj_id, obj in objects.items():
                        digest = obj.GetDigest() if hasattr(obj, "GetDigest") else {}
                        obj_type = digest.get("type", "unknown")
                        path = digest.get("path", "")
                        lines.append(f"  {obj_id}: {obj_type} {path}")
                    return True, "\n".join(lines)
                else:
                    return True, "No context objects."
            return True, "Session not found."

        elif command == "/dashboard":
            args = parts[1] if len(parts) > 1 else "open"
            subcommand_parts = args.split(maxsplit=1)
            subcommand = subcommand_parts[0].lower()

            from activecontext.dashboard import (
                get_dashboard_status,
                is_dashboard_running,
                start_dashboard,
                stop_dashboard,
            )

            if subcommand == "start":
                port = 8765  # default
                if len(subcommand_parts) > 1:
                    try:
                        port = int(subcommand_parts[1])
                    except ValueError:
                        return True, f"Invalid port: {subcommand_parts[1]}"

                if is_dashboard_running():
                    status = get_dashboard_status()
                    return True, f"Dashboard already running at http://127.0.0.1:{status['port']}"

                try:
                    await start_dashboard(
                        port=port,
                        manager=self._manager,
                        get_current_model=lambda: self._current_model_id,
                        sessions_model=self._sessions_model,
                        sessions_mode=self._sessions_mode,
                    )
                    return True, f"Dashboard started at http://127.0.0.1:{port}"
                except Exception as e:
                    return True, f"Failed to start dashboard: {e}"

            elif subcommand == "stop":
                if not is_dashboard_running():
                    return True, "Dashboard is not running."
                await stop_dashboard()
                return True, "Dashboard stopped."

            elif subcommand == "status":
                if is_dashboard_running():
                    status = get_dashboard_status()
                    return True, (
                        f"Dashboard running at http://127.0.0.1:{status['port']}\n"
                        f"  Uptime: {status['uptime']:.1f}s\n"
                        f"  Connections: {status['connections']}"
                    )
                return True, "Dashboard is not running."

            elif subcommand == "open":
                import webbrowser

                if not is_dashboard_running():
                    # Auto-start with default port
                    try:
                        await start_dashboard(
                            port=8765,
                            manager=self._manager,
                            get_current_model=lambda: self._current_model_id,
                            sessions_model=self._sessions_model,
                            sessions_mode=self._sessions_mode,
                        )
                    except Exception as e:
                        return True, f"Failed to start dashboard: {e}"

                status = get_dashboard_status()
                url = f"http://127.0.0.1:{status['port']}"
                webbrowser.open(url)
                return True, f"Opening dashboard at {url}"

            else:
                return True, (
                    "Usage: /dashboard [subcommand]\n"
                    "  start [port] - Start dashboard server (default port: 8765)\n"
                    "  stop         - Stop dashboard server\n"
                    "  status       - Show dashboard status\n"
                    "  open         - Open dashboard in browser (auto-starts if needed)"
                )

        # Unknown command - let it pass through to LLM
        return False, ""

    async def _emit_update(self, session_id: str, update: Any) -> None:
        """Convert and emit a SessionUpdate as an ACP notification."""
        if not self._conn:
            return

        # Priority flush: non-RESPONSE_CHUNK updates flush any pending chunks first
        if (
            self._batch_enabled
            and update.kind != UpdateKind.RESPONSE_CHUNK
            and session_id in self._chunk_buffers
        ):
            if session_id in self._flush_tasks:
                self._flush_tasks[session_id].cancel()
            await self._flush_chunks(session_id)

        match update.kind:
            case UpdateKind.STATEMENT_EXECUTING:
                # Emit as agent thought (shows as "thinking")
                source = update.payload.get("source", "")
                # Truncate long statements for display
                display = source[:100] + "..." if len(source) > 100 else source
                await self._conn.session_update(
                    session_id,
                    acp.update_agent_thought_text(f"Executing: {display}"),
                )

            case UpdateKind.STATEMENT_EXECUTED:
                status = update.payload.get("status", "ok")
                stdout = update.payload.get("stdout", "")
                exception = update.payload.get("exception")

                # Emit result as thought
                if status == "ok":
                    if stdout:
                        await self._conn.session_update(
                            session_id,
                            acp.update_agent_thought_text(f"Result: {stdout[:200]}"),
                        )
                else:
                    # Show error in thought
                    err_msg = exception.get("message", "Unknown error") if exception else "Error"
                    await self._conn.session_update(
                        session_id,
                        acp.update_agent_thought_text(f"Error: {err_msg}"),
                    )

            case UpdateKind.RESPONSE_CHUNK:
                text = update.payload.get("text", "")
                if text:
                    if self._batch_enabled:
                        await self._buffer_chunk(session_id, text)
                    else:
                        await self._conn.session_update(
                            session_id,
                            acp.update_agent_message_text(text),
                        )

            case UpdateKind.PROJECTION_READY:
                # Could emit as agent thought
                handles = update.payload.get("handles", {})
                if handles:
                    await self._conn.session_update(
                        session_id,
                        acp.update_agent_thought_text(
                            f"Context: {len(handles)} handles"
                        ),
                    )

        # Broadcast to dashboard if running
        from activecontext.dashboard import broadcast_update, is_dashboard_running

        if is_dashboard_running():
            await broadcast_update(
                session_id,
                update.kind.value,
                update.payload,
                update.timestamp,
            )


def create_agent() -> ActiveContextAgent:
    """Create a new ActiveContext ACP agent."""
    return ActiveContextAgent()
