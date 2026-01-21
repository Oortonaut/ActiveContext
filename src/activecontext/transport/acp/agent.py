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
    AvailableCommand,
    AvailableCommandInput,
    AvailableCommandsUpdate,
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
    UnstructuredCommandInput,
)

from activecontext.core.llm import (
    LiteLLMProvider,
    get_available_models,
    get_default_model,
)
from activecontext.logging import get_logger


def _find_jetbrains_chat_uuid() -> str | None:
    """Find JetBrains IDE chat UUID from its task history filesystem.

    JetBrains IDEs store chat history in aia-task-history/*.events files.
    The filename is the chat UUID. We find the most recently modified
    one to determine which chat is currently being used.

    This is a workaround for JetBrains IDEs not passing chat UUID in session/new.
    """
    import os
    from pathlib import Path

    # Find JetBrains task history directory
    localappdata = os.environ.get("LOCALAPPDATA")
    if not localappdata:
        return None

    jetbrains_dir = Path(localappdata) / "JetBrains"
    if not jetbrains_dir.exists():
        return None

    # Look for any JetBrains IDE installations (newest first)
    # Covers Rider, IntelliJ, PyCharm, WebStorm, etc.
    ide_dirs = sorted(
        [d for d in jetbrains_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,  # Most recently modified first
    )

    for ide_dir in ide_dirs:
        history_dir = ide_dir / "aia-task-history"
        if not history_dir.exists():
            continue

        # Find most recently modified .events file
        events_files = list(history_dir.glob("*.events"))
        if not events_files:
            continue

        # Sort by modification time, newest first
        events_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        newest = events_files[0]

        # Extract UUID from filename (e.g., "dc96b351-e6a6-48a8-8c32-c8458c7bc4b1.events")
        chat_uuid = newest.stem

        # Validate it looks like a UUID
        if len(chat_uuid) == 36 and chat_uuid.count("-") == 4:
            log.debug("Found JetBrains chat UUID: %s", chat_uuid)
            return chat_uuid

    return None
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
        self._chunk_lock = asyncio.Lock()  # Protects chunk buffer operations
        self._batch_enabled = True  # Can be disabled via config
        self._flush_interval = 0.05  # 50ms
        self._flush_threshold = 100  # characters
        self._closed_sessions: set[str] = set()  # Sessions that have been cancelled/closed
        self._active_prompts: dict[str, asyncio.Task[Any]] = {}  # session_id -> prompt task
        self._sessions_initialized: set[str] = set()  # Sessions that received post-setup
        self._agent_loop_tasks: dict[str, asyncio.Task[Any]] = {}  # session_id -> loop task
        self._load_batch_config()

        # Message completion tracking for prompt synchronization
        self._message_complete_events: dict[str, asyncio.Event] = {}

        # ACP update mode configuration
        # out_of_band_update=True: send updates immediately (async model)
        # out_of_band_update=False: queue updates between prompts (sync model)
        self._out_of_band_update = False
        self._load_acp_config()
        self._queued_updates: dict[str, list[Any]] = {}  # session_id -> list of updates
        self._in_prompt: set[str] = set()  # Sessions currently processing a prompt

        # ACP client info (populated during initialize handshake)
        self._client_info: dict[str, Any] | None = None
        self._protocol_version: int | None = None

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

    def _load_acp_config(self) -> None:
        """Load ACP transport settings from config."""
        try:
            from activecontext.config import get_config

            config = get_config()
            # Use the new typed ACPConfig
            self._out_of_band_update = config.acp.out_of_band_update
            log.debug("ACP out_of_band_update=%s", self._out_of_band_update)
        except ImportError:
            pass
        except Exception as e:
            log.debug("Failed to load ACP config: %s", e)

    # --- Nagle-style batching for RESPONSE_CHUNK ---

    async def _flush_chunks(self, session_id: str) -> None:
        """Flush accumulated response chunks for a session.

        Thread-safe: uses _chunk_lock to prevent races with cancel().
        """
        async with self._chunk_lock:
            # Check if session was closed while waiting for lock
            if session_id in self._closed_sessions:
                self._chunk_buffers.pop(session_id, None)
                self._flush_tasks.pop(session_id, None)
                return

            text = self._chunk_buffers.pop(session_id, "")
            self._flush_tasks.pop(session_id, None)

        # Send outside the lock to avoid holding it during I/O
        if text:
            await self._send_session_update(
                session_id,
                acp.update_agent_message_text(text),
            )

    async def _buffer_chunk(self, session_id: str, text: str) -> None:
        """Buffer a response chunk, flushing if threshold reached.

        Thread-safe: uses _chunk_lock to prevent races with cancel().
        """
        async with self._chunk_lock:
            # Check if session was closed - discard chunk
            if session_id in self._closed_sessions:
                return

            # Append to buffer
            self._chunk_buffers[session_id] = self._chunk_buffers.get(session_id, "") + text
            buffer_len = len(self._chunk_buffers[session_id])

        # Check size threshold - flush immediately if exceeded
        if buffer_len >= self._flush_threshold:
            async with self._chunk_lock:
                if session_id in self._flush_tasks:
                    self._flush_tasks[session_id].cancel()
                    self._flush_tasks.pop(session_id, None)
            await self._flush_chunks(session_id)
            return

        # Schedule flush if not already scheduled
        async with self._chunk_lock:
            if session_id not in self._flush_tasks and session_id not in self._closed_sessions:
                self._flush_tasks[session_id] = asyncio.create_task(
                    self._delayed_flush(session_id)
                )

    async def _delayed_flush(self, session_id: str) -> None:
        """Flush after delay (Nagle timer)."""
        try:
            await asyncio.sleep(self._flush_interval)
            await self._flush_chunks(session_id)
        except asyncio.CancelledError:
            # Expected when session is cancelled or threshold flush happens
            pass

    async def _cleanup_session_buffers(self, session_id: str) -> None:
        """Clean up batching state for a closed session.

        Thread-safe: uses _chunk_lock to prevent races with buffer/flush.
        """
        async with self._chunk_lock:
            self._chunk_buffers.pop(session_id, None)
            task = self._flush_tasks.pop(session_id, None)
            if task and not task.done():
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
        # Store client info for dashboard
        self._protocol_version = protocol_version
        self._client_info = {
            "name": client_info.name if client_info else "unknown",
            "version": client_info.version if client_info else "unknown",
            "title": getattr(client_info, "title", None) if client_info else None,
            "capabilities": None,
        }
        if client_capabilities:
            # Build flat list of all capabilities for dashboard display
            caps_list: list[dict[str, Any]] = []

            # Terminal capability
            terminal = getattr(client_capabilities, "terminal", False)
            caps_list.append({
                "name": "terminal",
                "label": "Terminal",
                "enabled": bool(terminal),
                "description": "Execute terminal commands",
            })

            # File system capabilities
            fs = getattr(client_capabilities, "fs", None)
            if fs:
                caps_list.append({
                    "name": "fs.read_text_file",
                    "label": "File Read",
                    "enabled": bool(getattr(fs, "read_text_file", False)),
                    "description": "Read text files",
                })
                caps_list.append({
                    "name": "fs.write_text_file",
                    "label": "File Write",
                    "enabled": bool(getattr(fs, "write_text_file", False)),
                    "description": "Write text files",
                })
                # Include fs._meta extensions
                fs_meta = getattr(fs, "field_meta", None)
                if fs_meta:
                    for key, value in fs_meta.items():
                        caps_list.append({
                            "name": f"fs._meta.{key}",
                            "label": key,
                            "enabled": bool(value) if isinstance(value, bool) else True,
                            "value": value if not isinstance(value, bool) else None,
                            "description": f"Extension: {key}",
                        })

            # Include top-level _meta extensions
            meta = getattr(client_capabilities, "field_meta", None)
            if meta:
                for key, value in meta.items():
                    caps_list.append({
                        "name": f"_meta.{key}",
                        "label": key,
                        "enabled": bool(value) if isinstance(value, bool) else True,
                        "value": value if not isinstance(value, bool) else None,
                        "description": f"Extension: {key}",
                    })

            self._client_info["capabilities"] = caps_list

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

    def get_client_info(self) -> tuple[dict[str, Any] | None, int | None]:
        """Get ACP client information for dashboard.

        Returns:
            Tuple of (client_info dict, protocol_version).
            Both are None if initialize() hasn't been called yet.
        """
        return self._client_info, self._protocol_version

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp.NewSessionResponse:
        """Create a new session, or resume existing one if JetBrains UUID matches."""
        import traceback

        try:
            # Track current cwd for list_sessions
            self._current_cwd = cwd

            # Try to find JetBrains IDE chat UUID from filesystem
            # This allows session resumption even though JetBrains doesn't call session/load
            jetbrains_uuid = _find_jetbrains_chat_uuid()
            session = None

            if jetbrains_uuid:
                # Check if already in memory (other hosts might not restart agent)
                existing = await self._manager.get_session(jetbrains_uuid)
                if existing:
                    log.info("Resuming session %s (in memory)", jetbrains_uuid)
                    session = existing
                else:
                    # Try loading from disk
                    loaded = Session.from_file(
                        cwd=cwd,
                        session_id=jetbrains_uuid,
                        llm=self._manager._default_llm,
                    )
                    if loaded:
                        log.info("Resuming session %s (from disk)", jetbrains_uuid)
                        self._manager._sessions[jetbrains_uuid] = loaded
                        session = loaded

            if session is None:
                # Create new session with JetBrains UUID (or generate one)
                session = await self._manager.create_session(
                    cwd=cwd,
                    session_id=jetbrains_uuid,  # Will generate UUID if None
                    permission_requester=None,
                    shell_permission_requester=None,
                    website_permission_requester=None,
                    import_permission_requester=None,
                )
                log.info("Created session %s", session.session_id)

                # Save new session to disk immediately
                try:
                    session.save()
                    log.debug("Saved new session %s to disk", session.session_id)
                except Exception as e:
                    log.warning("Failed to save session to disk: %s", e)

            # Set up ACP terminal executor (needed for both new and resumed sessions)
            if self._conn:
                terminal_executor = ACPTerminalExecutor(
                    client=self._conn,
                    session_id=session.session_id,
                    default_cwd=cwd,
                )
                session.timeline._terminal_executor = terminal_executor

            # Track session metadata
            self._sessions_cwd[session.session_id] = cwd
            self._sessions_mode[session.session_id] = self._default_mode_id

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
            log.info("  Models (%d): %s", len(model_ids), ", ".join(model_ids) if model_ids else "none")
            log.info("  Modes (%d): %s", len(mode_ids), ", ".join(mode_ids))

            # Start the agent loop for async prompt processing
            await self._start_agent_loop(session)

            # Schedule post-session setup to run AFTER response is sent
            # (asyncio.create_task schedules but doesn't execute until we yield)
            if session.session_id not in self._sessions_initialized:
                self._sessions_initialized.add(session.session_id)
                asyncio.create_task(self._post_session_setup(session.session_id))

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
                # Ensure agent loop is running
                await self._start_agent_loop(existing)
                # Schedule post-session setup to run AFTER response is sent
                if session_id not in self._sessions_initialized:
                    self._sessions_initialized.add(session_id)
                    asyncio.create_task(self._post_session_setup(session_id))
                return acp.LoadSessionResponse(
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

            # Start the agent loop for async prompt processing
            await self._start_agent_loop(session)

            # Schedule post-session setup to run AFTER response is sent
            if session_id not in self._sessions_initialized:
                self._sessions_initialized.add(session_id)
                asyncio.create_task(self._post_session_setup(session_id))

            return acp.LoadSessionResponse(
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
        # Also update global current model so dashboard reflects the change
        self._current_model_id = model_id

        log.info("Model changed to %s for session %s", model_id, session_id)
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

        Behavior depends on out_of_band_update config:
        - True (async): Queue message, return immediately, send updates as notifications
        - False (sync): Queue message, wait for completion, then return

        When out_of_band_update=False, updates that occurred between prompts are
        flushed at the start of each prompt.

        Per ACP spec, every prompt must have exactly one PromptResponse.
        """
        import uuid

        # Check if session was already cancelled before we start
        if session_id in self._closed_sessions:
            log.info("Prompt received for already-cancelled session %s", session_id)
            return acp.PromptResponse(stop_reason="cancelled")

        session = await self._manager.get_session(session_id)
        if not session:
            log.error("Session not found in manager: %s", session_id)
            raise acp.RequestError(
                code=-32600,
                message=f"Session not found: {session_id}",
            )

        # Mark as in-prompt (for update queueing logic)
        self._in_prompt.add(session_id)

        try:
            # Flush any queued updates from between prompts
            await self._flush_queued_updates(session_id)

            # Extract text from prompt blocks
            content = ""
            for block in prompt:
                if isinstance(block, TextContentBlock) or hasattr(block, "text"):
                    content += block.text

            # Handle slash commands synchronously (they don't go through the queue)
            if content.strip().startswith("/"):
                handled, response = await self._handle_slash_command(
                    content.strip(), session_id
                )
                if handled:
                    if response:
                        await self._send_session_update(
                            session_id,
                            acp.update_agent_message_text(response),
                        )
                    return acp.PromptResponse(stop_reason="end_turn")

            # Generate message ID
            message_id = f"msg_{uuid.uuid4().hex[:8]}"

            # Queue the message (wakes agent loop)
            session.queue_user_message(content, message_id)
            log.info("Queued message %s for session %s", message_id, session_id)

            # Behavior depends on out_of_band_update mode
            if self._out_of_band_update:
                # Async mode: return immediately, processing happens in background
                return acp.PromptResponse(stop_reason="end_turn")

            # Sync mode: wait for message processing to complete
            completion_event = asyncio.Event()
            self._message_complete_events[message_id] = completion_event

            try:
                # Wait for message processing to complete (with timeout)
                try:
                    # Use a reasonable timeout (5 minutes for long LLM responses)
                    await asyncio.wait_for(completion_event.wait(), timeout=300.0)
                    log.info("Message %s completed for session %s", message_id, session_id)
                except asyncio.TimeoutError:
                    log.warning("Message %s timed out for session %s", message_id, session_id)
                except asyncio.CancelledError:
                    log.info("Message %s cancelled for session %s", message_id, session_id)
                    if session_id in self._closed_sessions:
                        return acp.PromptResponse(stop_reason="cancelled")
                    raise

                # Check if session was cancelled during processing
                if session_id in self._closed_sessions:
                    return acp.PromptResponse(stop_reason="cancelled")

                return acp.PromptResponse(stop_reason="end_turn")

            finally:
                # Clean up the completion event
                self._message_complete_events.pop(message_id, None)

        finally:
            # Clear in-prompt state
            self._in_prompt.discard(session_id)

    async def _process_prompt(
        self,
        session: Session,
        session_id: str,
        prompt: list[Any],
    ) -> acp.PromptResponse:
        """Internal prompt processing, separated for cleaner cancellation handling."""
        import traceback

        from activecontext.mcp.hooks import set_pre_call_hook

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
                if response:
                    await self._send_session_update(
                        session_id,
                        acp.update_agent_message_text(response),
                    )
                return acp.PromptResponse(stop_reason="end_turn")

        # Create MCP pre-call hook for UI feedback
        async def mcp_pre_call_hook(
            server_name: str, tool_name: str, arguments: dict[str, Any]
        ) -> None:
            """Send thought update before MCP tool invocation."""
            # Format arguments for display (truncate long values)
            args_preview = ", ".join(
                f"{k}={repr(v)[:50]}" for k, v in list(arguments.items())[:3]
            )
            if len(arguments) > 3:
                args_preview += ", ..."
            await self._send_session_update(
                session_id,
                acp.update_agent_thought_text(
                    f"Calling {server_name}.{tool_name}({args_preview})..."
                ),
            )

        # Process prompt and stream updates
        try:
            # Set the MCP pre-call hook for this prompt
            set_pre_call_hook(mcp_pre_call_hook)

            response_text = ""
            async for update in session.prompt(content):
                # Check if cancelled - break out early
                if session_id in self._closed_sessions:
                    log.info("Prompt cancelled for session %s", session_id)
                    break

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
        except asyncio.CancelledError:
            # Re-raise to be handled by prompt()
            raise
        except Exception as e:
            # Check if this exception is due to cancellation - per ACP spec,
            # we must return cancelled stop reason, not propagate errors
            if session_id in self._closed_sessions:
                log.info("Exception during cancelled session %s: %s", session_id, e)
                return acp.PromptResponse(stop_reason="cancelled")

            # Log the error but don't crash the agent
            log.error("Error in prompt: %s", e)
            log.error("%s", traceback.format_exc())
            # Send error message to user
            await self._send_session_update(
                session_id,
                acp.update_agent_message_text(f"\n\nError: {e}"),
            )
        finally:
            # Always clear the MCP pre-call hook
            set_pre_call_hook(None)

        # Check if session was cancelled - return cancelled stop reason per ACP spec
        if session_id in self._closed_sessions:
            log.info("Returning cancelled for session %s", session_id)
            return acp.PromptResponse(stop_reason="cancelled")

        # Auto-save session after each prompt (skip for cancelled sessions)
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
        """Cancel the current in-progress prompt in a session.

        This is called when Rider sends session/cancel to interrupt an
        ongoing prompt (e.g., user clicks stop button during generation).

        Note: session/cancel is ONLY for cancelling prompts, NOT for session
        termination. Session teardown happens via process termination.

        Per ACP spec, if there's an active prompt, it must return with
        stop_reason="cancelled". We track active prompts and cancel them.

        IMPORTANT: This is a notification handler - it must complete quickly
        and not block waiting for external resources.
        """
        log.info("Session cancel received: %s", session_id)

        # Mark session as closed first - prevents new updates from being sent
        self._closed_sessions.add(session_id)
        log.debug("Cancel [%s]: marked as closed", session_id)

        # Cancel any active prompt task for this session
        prompt_task = self._active_prompts.get(session_id)
        if prompt_task and not prompt_task.done():
            log.info("Cancelling active prompt task for session %s", session_id)
            prompt_task.cancel()
            try:
                # Wait briefly for task to acknowledge cancellation
                # Use shield to prevent our wait from being cancelled
                await asyncio.wait_for(asyncio.shield(prompt_task), timeout=1.0)
                log.debug("Cancel [%s]: prompt task completed", session_id)
            except asyncio.CancelledError:
                log.debug("Cancel [%s]: prompt task acknowledged cancellation", session_id)
            except asyncio.TimeoutError:
                log.warning("Cancel [%s]: prompt task did not complete within 1s", session_id)
            except Exception as e:
                log.warning("Cancel [%s]: error waiting for prompt task: %s", session_id, e)
        else:
            log.debug("Cancel [%s]: no active prompt task", session_id)

        # Clean up chunk buffers with timeout to avoid blocking
        log.debug("Cancel [%s]: cleaning up chunk buffers", session_id)
        try:
            await asyncio.wait_for(self._cleanup_session_buffers(session_id), timeout=1.0)
            log.debug("Cancel [%s]: chunk buffers cleaned", session_id)
        except asyncio.TimeoutError:
            log.warning("Cancel [%s]: chunk buffer cleanup timed out", session_id)
            # Force cleanup without lock if timeout
            self._chunk_buffers.pop(session_id, None)
            task = self._flush_tasks.pop(session_id, None)
            if task:
                task.cancel()

        # Cancel the session in the manager
        log.debug("Cancel [%s]: getting session from manager", session_id)
        session = await self._manager.get_session(session_id)
        if session:
            log.debug("Cancel [%s]: cancelling session in manager", session_id)
            await session.cancel()
            log.debug("Cancel [%s]: session cancelled in manager", session_id)

            # Check if cancellation propagated to the task
            if session._current_task and not session._current_task.done():
                log.warning(
                    "Session %s: _current_task still running after cancel() - "
                    "cancellation may not have propagated",
                    session_id,
                )
        else:
            log.warning("Session %s not found in manager during cancel", session_id)

        # Clean up session tracking (only for sessions we created, not the main IDE session)
        # This prevents unbounded growth of _closed_sessions for child agent sessions
        self._cleanup_closed_session(session_id)
        log.info("Cancel [%s]: completed", session_id)

    def _cleanup_closed_session(self, session_id: str) -> None:
        """Clean up tracking state for a fully closed session.

        Called after cancel() completes to remove the session from tracking sets.
        This is safe because:
        1. The session is already marked as closed in _closed_sessions
        2. Any prompt will return cancelled (checked at start of prompt())
        3. Chunk buffers are cleaned up
        """
        # Clean up metadata tracking
        self._sessions_cwd.pop(session_id, None)
        self._sessions_model.pop(session_id, None)
        self._sessions_mode.pop(session_id, None)

        # Remove from closed sessions set to prevent unbounded growth
        # This is safe because:
        # - The session is cancelled in the manager
        # - Any new prompt for this session will get "session not found" error
        self._closed_sessions.discard(session_id)

        log.debug("Cleaned up session tracking for %s", session_id)

    async def ext_method(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle extension methods."""
        log.debug("Received extension method: %s", method)
        return {}

    async def ext_notification(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        """Handle extension notifications."""
        log.debug("Received extension notification: %s", method)

    async def _handle_slash_command(
        self, content: str, session_id: str
    ) -> tuple[bool, str]:
        """Handle slash commands before they reach the LLM.

        Returns:
            (handled, response) - handled=True if command was processed
        """
        parts = content.split(maxsplit=1)
        command = parts[0].lower()
        # args = parts[1] if len(parts) > 1 else ""

        if command == "/help":
            return True, (
                "Available commands:\n"
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
                session.clear_message_history()
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
                        get_client_info=self.get_client_info,
                        transport_type="acp",
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
                            get_client_info=self.get_client_info,
                            transport_type="acp",
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

    async def _send_session_update(self, session_id: str, update: Any) -> None:
        """Send a session update, checking if session is still open."""
        if not self._conn or session_id in self._closed_sessions:
            return
        await self._conn.session_update(session_id, update)

    async def _post_session_setup(self, session_id: str) -> None:
        """Post-session setup hook called after session is created or loaded."""
        try:
            # Advertise available slash commands to client
            await self._send_session_update(
                session_id,
                AvailableCommandsUpdate(
                    session_update="available_commands_update",
                    available_commands=self._get_available_commands(),
                ),
            )
            log.debug("Sent available commands for session %s", session_id)
        except Exception as e:
            log.warning("Failed to send available commands for session %s: %s", session_id, e)

    def _get_available_commands(self) -> list[AvailableCommand]:
        """Build list of available slash commands for ACP clients."""
        return [
            AvailableCommand(
                name="help",
                description="Show available commands",
            ),
            AvailableCommand(
                name="clear",
                description="Clear conversation history",
            ),
            AvailableCommand(
                name="context",
                description="Show current context objects",
            ),
            AvailableCommand(
                name="title",
                description="Get or set session title",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="New title (optional)")
                ),
            ),
            AvailableCommand(
                name="dashboard",
                description="Open monitoring dashboard",
                input=AvailableCommandInput(
                    root=UnstructuredCommandInput(hint="start|stop|status|open")
                ),
            ),
        ]

    async def _flush_queued_updates(self, session_id: str) -> None:
        """Flush any queued updates for a session.

        Called at the start of prompt() to send updates that accumulated
        between prompts when out_of_band_update=False.
        """
        if session_id not in self._queued_updates:
            return

        queued = self._queued_updates.pop(session_id)
        if not queued:
            return

        log.debug("Flushing %d queued updates for session %s", len(queued), session_id)
        for update in queued:
            await self._emit_update_internal(session_id, update)

    def _queue_update(self, session_id: str, update: Any) -> None:
        """Queue an update for later delivery."""
        if session_id not in self._queued_updates:
            self._queued_updates[session_id] = []
        self._queued_updates[session_id].append(update)
        log.debug("Queued update %s for session %s", update.kind, session_id)

    async def _emit_update(self, session_id: str, update: Any) -> None:
        """Convert and emit a SessionUpdate as an ACP notification.

        When out_of_band_update=False and not in a prompt, queues the update
        for delivery when the next prompt arrives.
        """
        # Check if we should queue this update
        if not self._out_of_band_update and session_id not in self._in_prompt:
            self._queue_update(session_id, update)
            # Still broadcast to dashboard even when queueing
            from activecontext.dashboard import broadcast_update, is_dashboard_running
            if is_dashboard_running():
                await broadcast_update(
                    session_id,
                    update.kind.value,
                    update.payload,
                    update.timestamp,
                )
            return

        await self._emit_update_internal(session_id, update)

    async def _emit_update_internal(self, session_id: str, update: Any) -> None:
        """Internal method to convert and emit a SessionUpdate as an ACP notification."""
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
                await self._send_session_update(
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
                        await self._send_session_update(
                            session_id,
                            acp.update_agent_thought_text(f"Result: {stdout[:200]}"),
                        )
                else:
                    # Show error in thought
                    err_msg = exception.get("message", "Unknown error") if exception else "Error"
                    await self._send_session_update(
                        session_id,
                        acp.update_agent_thought_text(f"Error: {err_msg}"),
                    )

            case UpdateKind.RESPONSE_CHUNK:
                text = update.payload.get("text", "")
                if text:
                    if self._batch_enabled:
                        await self._buffer_chunk(session_id, text)
                    else:
                        await self._send_session_update(
                            session_id,
                            acp.update_agent_message_text(text),
                        )

            case UpdateKind.PROJECTION_READY:
                # Could emit as agent thought
                handles = update.payload.get("handles", {})
                if handles:
                    await self._send_session_update(
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

    async def _start_agent_loop(self, session: Session) -> None:
        """Start the agent loop for a session.

        The agent loop runs in the background and processes:
        - Queued user messages
        - File change events
        - MCP async results
        - Other wake events

        Args:
            session: The session to start the loop for
        """
        session_id = session.session_id

        # Don't start if already running
        if session_id in self._agent_loop_tasks:
            existing = self._agent_loop_tasks[session_id]
            if not existing.done():
                log.debug("Agent loop already running for session %s", session_id)
                return

        async def _run_loop() -> None:
            """Wrapper to run the agent loop and emit updates."""
            try:
                async for update in session.run_agent_loop():
                    # Check if session was closed
                    if session_id in self._closed_sessions:
                        log.info("Agent loop stopping for closed session %s", session_id)
                        break

                    if self._conn:
                        await self._emit_update(session_id, update)

                    # Signal message completion when PROJECTION_READY
                    if update.kind == UpdateKind.PROJECTION_READY:
                        message_id = update.payload.get("message_id")
                        if message_id and message_id in self._message_complete_events:
                            self._message_complete_events[message_id].set()
                            log.debug("Signaled completion for message %s", message_id)

                        # Auto-save after each message completion
                        try:
                            session.save()
                            log.debug("Auto-saved session %s", session_id)
                        except Exception as e:
                            log.warning("Failed to auto-save session: %s", e)

            except asyncio.CancelledError:
                log.info("Agent loop cancelled for session %s", session_id)
            except Exception as e:
                log.error("Agent loop error for session %s: %s", session_id, e)
                import traceback
                log.error("%s", traceback.format_exc())
            finally:
                self._agent_loop_tasks.pop(session_id, None)
                log.info("Agent loop ended for session %s", session_id)

        # Start the loop as a background task
        task = asyncio.create_task(_run_loop())
        self._agent_loop_tasks[session_id] = task
        log.info("Started agent loop for session %s", session_id)

    def _stop_agent_loop(self, session_id: str) -> None:
        """Stop the agent loop for a session.

        Args:
            session_id: The session whose loop to stop
        """
        task = self._agent_loop_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            log.info("Cancelled agent loop for session %s", session_id)


def create_agent() -> ActiveContextAgent:
    """Create a new ActiveContext ACP agent."""
    return ActiveContextAgent()
