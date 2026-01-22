"""Session and SessionManager implementations.

The SessionManager is the top-level entry point for both ACP and Direct transports.
Each Session wraps a Timeline and provides the high-level interface for prompts,
ticks, and projections.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from activecontext.context.graph import ContextGraph
from activecontext.context.nodes import (
    GroupNode,
    MCPManagerNode,
    MessageNode,
    SessionNode,
    TextNode,
)
from activecontext.context.state import Expansion, TickFrequency
from activecontext.core.projection_engine import ProjectionEngine
from activecontext.logging import get_logger
from activecontext.session.permissions import ImportGuard, PermissionManager, ShellPermissionManager
from activecontext.session.protocols import (
    Projection,
    SessionUpdate,
    UpdateKind,
)
from activecontext.session.storage import (
    generate_default_title,
    get_session_path,
    load_session_data,
    save_session,
)
from activecontext.session.timeline import Timeline

log = get_logger("session")

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

    from activecontext.config.schema import Config
    from activecontext.context.buffer import TextBuffer
    from activecontext.context.state import Notification
    from activecontext.core.llm.provider import LLMProvider, Message
    from activecontext.terminal.protocol import TerminalExecutor

    # Type for file permission requester callback:
    # async (session_id, path, mode) -> (granted, persist)
    PermissionRequester = Callable[
        [str, str, str], Coroutine[Any, Any, tuple[bool, bool]]
    ]

    # Type for shell permission requester callback:
    # async (session_id, command, args) -> (granted, persist)
    ShellPermissionRequester = Callable[
        [str, str, list[str] | None], Coroutine[Any, Any, tuple[bool, bool]]
    ]

    # Type for website permission requester callback:
    # async (session_id, url, method) -> (granted, persist)
    WebsitePermissionRequester = Callable[
        [str, str, str], Coroutine[Any, Any, tuple[bool, bool]]
    ]

    # Type for import permission requester callback:
    # async (session_id, module) -> (granted, persist, include_submodules)
    ImportPermissionRequester = Callable[
        [str, str], Coroutine[Any, Any, tuple[bool, bool, bool]]
    ]


class Session:
    """A session wrapping a timeline with prompt and tick handling.

    Implements SessionProtocol.
    """

    def __init__(
        self,
        session_id: str,
        cwd: str,
        timeline: Timeline,
        llm: LLMProvider | None = None,
        config: Config | None = None,
        title: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self._session_id = session_id
        self._cwd = cwd
        self._timeline = timeline
        self._llm = llm
        self._config = config
        self._cancelled = False
        self._current_task: asyncio.Task[Any] | None = None
        self._message_history: list[Message] = []
        self._projection_engine = self._build_projection_engine(config)

        # Session persistence metadata
        self._title = title or generate_default_title()
        self._created_at = created_at or datetime.now()

        # Track timing for turn statistics
        self._turn_start_time: float | None = None

        # Context stack for structural containment
        # When non-empty, new nodes are added as children of stack.top
        self._context_stack: list[str] = []

        # Text buffer storage - Session owns this, timeline references it
        self._text_buffers: dict[str, TextBuffer] = {}
        self._timeline._text_buffers = self._text_buffers

        # Check if we're restoring from a saved session (context graph already has nodes)
        existing_context = self._timeline.context_graph.get_node("context")
        is_restore = existing_context is not None

        if is_restore:
            # Restoring: use the existing root context node
            if isinstance(existing_context, GroupNode):
                self._root_context = existing_context
            else:
                # Shouldn't happen, but create fresh if somehow wrong type
                self._root_context = GroupNode(
                    node_id="context",
                    expansion=Expansion.DETAILS,
                    mode="running",
                    tick_frequency=TickFrequency.turn(),
                )
                self._root_context.is_subscription_point = True
                self._timeline.context_graph.add_node(self._root_context)
                self._timeline.context_graph.set_root("context")
        else:
            # Fresh session: create root Context node for document-ordered rendering
            # All other nodes become children of this root
            self._root_context = GroupNode(
                node_id="context",
                expansion=Expansion.DETAILS,
                mode="running",
                tick_frequency=TickFrequency.turn(),
            )
            # Root is auto-subscribed to collect all notifications from subtree
            self._root_context.is_subscription_point = True
            self._timeline.context_graph.add_node(self._root_context)
            self._timeline.context_graph.set_root("context")

        self._context_stack.append("context")  # All nodes go inside root
        self._timeline.set_current_group("context")

        # Add system prompt only for fresh sessions (restored sessions have it)
        if not is_restore:
            self._add_system_prompt_node()

        # SessionNode and MCPManagerNode are created in _create_metadata_nodes()
        # Called after context guide to maintain document order:
        # System Prompt -> Guide -> Session -> MCP -> Messages
        self._session_node: SessionNode | None = None
        self._mcp_manager_node: MCPManagerNode | None = None

        # Register set_title callback with timeline
        self._timeline.set_title_callback(self.set_title)

        # Register path resolver callback with timeline for @prompts/ etc.
        self._timeline._path_resolver = self.resolve_path

        # Configure file watcher from config
        if self._config and self._config.session:
            self._timeline.configure_file_watcher(self._config.session.file_watch)

        # Event-driven agent loop infrastructure
        self._wake_event = asyncio.Event()
        self._running = False
        self._agent_task: asyncio.Task[Any] | None = None

        # User messages group - created in _create_metadata_nodes
        self._user_messages_group: GroupNode | None = None

        # Alerts group for notifications - created in _create_metadata_nodes
        self._alerts_group: GroupNode | None = None

    def _add_system_prompt_node(self) -> None:
        """Add the system prompt as a fully expanded TextNode tree.

        The system prompt is parsed into a TextNode tree and added
        to the context graph with state=ALL so it renders fully expanded.
        Links to root context for document ordering.

        Note: Only the base system prompt is loaded here. Reference prompts
        (dsl_reference, node_states, etc.) are loaded via startup statements.
        """
        from activecontext.prompts import SYSTEM_PROMPT

        # Use timeline's markdown parsing to create TextNode tree
        root = self._timeline._make_markdown_node(
            path="system_prompt",
            content=SYSTEM_PROMPT,
            tokens=2000,  # Base system prompt is smaller than full combined prompt
            expansion=Expansion.DETAILS,  # Fully expanded
        )

        # Link root node to root context for document ordering
        self._timeline.context_graph.link(root.node_id, "context")

    def _restore_system_prompt_buffer(self) -> None:
        """Restore text buffer for system prompt nodes after session load.

        System prompt content comes from SYSTEM_PROMPT constant, not from
        a file. After restoring from disk, the TextNodes exist but their buffer
        reference is lost. This method recreates the buffer and re-links the nodes.

        Also handles @prompts/ paths which are resolved via resolve_path().
        """
        from activecontext.context.buffer import TextBuffer
        from activecontext.prompts import SYSTEM_PROMPT

        # Track paths we need to restore (path -> nodes using that path)
        paths_to_restore: dict[str, list[TextNode]] = {}

        for node in self._timeline.context_graph:
            if isinstance(node, TextNode):
                if node.path == "system_prompt":
                    paths_to_restore.setdefault("system_prompt", []).append(node)
                elif node.path.startswith("@prompts/"):
                    paths_to_restore.setdefault(node.path, []).append(node)

        # Restore system_prompt buffer
        if "system_prompt" in paths_to_restore:
            buffer = TextBuffer(
                path="system_prompt",
                lines=SYSTEM_PROMPT.split("\n"),
            )
            self._text_buffers[buffer.buffer_id] = buffer
            for node in paths_to_restore["system_prompt"]:
                node.buffer_id = buffer.buffer_id

        # Restore @prompts/ buffers using resolve_path
        for path, nodes in paths_to_restore.items():
            if path.startswith("@prompts/"):
                resolved_path, content = self.resolve_path(path)
                if content is not None:
                    buffer = TextBuffer(
                        path=resolved_path,
                        lines=content.split("\n"),
                    )
                    self._text_buffers[buffer.buffer_id] = buffer
                    for node in nodes:
                        node.buffer_id = buffer.buffer_id
            # Note: start_line/end_line should be preserved from saved node
            # If they're default values (1/None), the node renders the whole buffer

    def _create_metadata_nodes(self) -> None:
        """Create SessionNode and MCPManagerNode.

        Called after context guide is loaded to maintain document order:
        System Prompt -> Guide -> Session -> MCP -> Messages
        """
        # Create SessionNode for agent situational awareness
        self._session_node = SessionNode(
            node_id="session",
            tokens=500,
            mode="running",
            tick_frequency=TickFrequency.turn(),
            session_start_time=self._created_at.timestamp(),
        )
        self._timeline.context_graph.add_node(self._session_node)
        self._timeline.context_graph.link("session", "context")

        # Create MCPManagerNode singleton for tracking MCP server connections
        self._mcp_manager_node = MCPManagerNode(
            node_id="mcp_manager",
            tokens=300,
            expansion=Expansion.SUMMARY,
            mode="running",
            tick_frequency=TickFrequency.turn(),
        )
        self._timeline.context_graph.add_node(self._mcp_manager_node)
        self._timeline.context_graph.link("mcp_manager", "context")

        # Create User Messages group for queued async messages
        # Document order: System Prompt -> Guide -> Session -> MCP -> User Messages
        self._user_messages_group = GroupNode(
            node_id="user_messages",
            tokens=2000,
            expansion=Expansion.DETAILS,  # Visible in projection
            mode="running",
            tick_frequency=TickFrequency.turn(),
        )
        self._timeline.context_graph.add_node(self._user_messages_group)
        self._timeline.context_graph.link("user_messages", "context")

        # Create Alerts group for notifications
        # Notifications from HOLD/WAKE level nodes appear here
        self._alerts_group = GroupNode(
            node_id="alerts",
            tokens=500,
            expansion=Expansion.HIDDEN,  # Hidden when empty, DETAILS when has content
            mode="running",
            tick_frequency=TickFrequency.turn(),
        )
        self._timeline.context_graph.add_node(self._alerts_group)
        self._timeline.context_graph.link("alerts", "context")

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def llm(self) -> LLMProvider | None:
        return self._llm

    def set_llm(self, llm: LLMProvider | None) -> None:
        """Set or update the LLM provider."""
        self._llm = llm

    @property
    def title(self) -> str:
        """Session title for display in IDE menus."""
        return self._title

    def set_title(self, title: str) -> None:
        """Set the session title.

        Args:
            title: New title for the session.
        """
        self._title = title

    def resolve_path(self, path: str) -> tuple[str, str | None]:
        """Resolve path prefixes and roots to actual paths or content.

        Supports special path prefixes and roots for cross-platform compatibility:

        **Content prefixes** (return path + content):
        - @prompts/: Bundled reference prompts (e.g., @prompts/dsl_reference.md)

        **Path roots** (return resolved path, no content):
        - ~ : User home directory (Unix style)
        - {home}: User home directory
        - $HOME: User home directory (Unix env var style)
        - %USERPROFILE%: User home directory (Windows env var style)
        - {cwd}: Session working directory
        - $CWD: Session working directory
        - {PROJECT}: Project root (same as cwd)

        Path separators are normalized to the platform's native separator.

        Args:
            path: File path, possibly with a prefix or root

        Returns:
            Tuple of (resolved_path, content_or_none):
            - For @prompts/: returns (name, prompt_content)
            - For path roots: returns (absolute_path, None)
            - For regular paths: returns (path, None)
        """
        import os

        # Handle @prompts/ prefix (content provider)
        if path.startswith("@prompts/"):
            # Extract prompt name: "@prompts/dsl_reference.md" -> "dsl_reference"
            name = path[9:]  # Remove "@prompts/"
            if name.endswith(".md"):
                name = name[:-3]  # Remove ".md"
            from activecontext.prompts import load_prompt

            content = load_prompt(name)
            return (f"@prompts/{name}", content)

        # Path root expansion (returns resolved path, no content)
        resolved = self._expand_path_roots(path)

        # Normalize separators to platform native
        resolved = os.path.normpath(resolved)

        return (resolved, None)

    def _expand_path_roots(self, path: str) -> str:
        """Expand path root prefixes to absolute paths.

        Supports cross-platform path roots:
        - ~ : User home directory (tilde expansion)
        - {home}: User home directory (brace syntax)
        - $HOME: User home directory (Unix env var syntax)
        - %USERPROFILE%: User home directory (Windows env var syntax)
        - {cwd}: Session working directory
        - $CWD: Session working directory
        - {PROJECT}: Project root (same as cwd)

        Args:
            path: Path that may start with a root prefix

        Returns:
            Path with root prefix expanded to absolute path
        """
        import os

        # Get home and cwd for expansion
        home = os.path.expanduser("~")
        cwd = self._cwd

        def strip_leading_seps(s: str) -> str:
            """Strip all leading path separators from a string."""
            while s and s[0] in "/\\":
                s = s[1:]
            return s

        def is_path_boundary(s: str) -> bool:
            """Check if string is empty or starts with a path separator."""
            return not s or s[0] in "/\\"

        # Home directory roots - handle in order of specificity
        # Windows-style: %USERPROFILE%/... or %USERPROFILE%\...
        if path.upper().startswith("%USERPROFILE%"):
            rest = path[13:]  # len("%USERPROFILE%")
            if is_path_boundary(rest):
                rest = strip_leading_seps(rest)
                return os.path.join(home, rest) if rest else home

        # Unix-style: $HOME/... or $HOME\... (must be followed by separator or end)
        if path.startswith("$HOME"):
            rest = path[5:]  # len("$HOME")
            if is_path_boundary(rest):
                rest = strip_leading_seps(rest)
                return os.path.join(home, rest) if rest else home

        # Brace-style: {home}/... or {home}\...
        if path.lower().startswith("{home}"):
            rest = path[6:]  # len("{home}")
            if is_path_boundary(rest):
                rest = strip_leading_seps(rest)
                return os.path.join(home, rest) if rest else home

        # Tilde: ~/... or ~\... (must be followed by separator or end)
        # Note: ~username is a valid Unix path, so we only match ~ followed by / or \ or end
        if path.startswith("~"):
            rest = path[1:]
            if is_path_boundary(rest):
                rest = strip_leading_seps(rest)
                return os.path.join(home, rest) if rest else home

        # CWD/Project roots
        # Brace-style: {cwd}/... or {PROJECT}/...
        if path.lower().startswith("{cwd}"):
            rest = path[5:]  # len("{cwd}")
            if is_path_boundary(rest):
                rest = strip_leading_seps(rest)
                return os.path.join(cwd, rest) if rest else cwd

        if path.lower().startswith("{project}"):
            rest = path[9:]  # len("{project}")
            if is_path_boundary(rest):
                rest = strip_leading_seps(rest)
                return os.path.join(cwd, rest) if rest else cwd

        # Unix-style: $CWD/... (must be followed by separator or end)
        if path.startswith("$CWD"):
            rest = path[4:]  # len("$CWD")
            if is_path_boundary(rest):
                rest = strip_leading_seps(rest)
                return os.path.join(cwd, rest) if rest else cwd

        # No recognized root - return as-is
        return path

    # -------------------------------------------------------------------------
    # Text Buffer Management
    # -------------------------------------------------------------------------

    def get_text_buffer(self, buffer_id: str) -> TextBuffer | None:
        """Get a TextBuffer by ID.

        Args:
            buffer_id: The buffer's unique identifier

        Returns:
            TextBuffer if found, None otherwise
        """
        return self._text_buffers.get(buffer_id)

    def add_text_buffer(self, buffer: TextBuffer) -> str:
        """Add a TextBuffer to the session.

        Args:
            buffer: The TextBuffer to add

        Returns:
            The buffer's ID
        """
        self._text_buffers[buffer.buffer_id] = buffer
        return buffer.buffer_id

    def get_or_create_text_buffer(self, path: str) -> TextBuffer:
        """Get an existing TextBuffer for a path, or create one.

        Args:
            path: File path to load

        Returns:
            Existing or newly created TextBuffer
        """
        from activecontext.context.buffer import TextBuffer

        # Check if we already have a buffer for this path
        for buffer in self._text_buffers.values():
            if buffer.path == path:
                return buffer

        # Create new buffer from file
        buffer = TextBuffer.from_file(path, cwd=self._cwd)
        self._text_buffers[buffer.buffer_id] = buffer
        return buffer

    @property
    def text_buffers(self) -> dict[str, TextBuffer]:
        """All text buffers in the session."""
        return self._text_buffers

    @property
    def created_at(self) -> datetime:
        """When the session was created."""
        return self._created_at

    # -------------------------------------------------------------------------
    # Context Stack Operations
    # -------------------------------------------------------------------------

    @property
    def current_group(self) -> str | None:
        """Get the current container group ID, or None for root level.

        When a group is on the stack, new nodes added via add_node()
        will automatically become children of that group.
        """
        return self._context_stack[-1] if self._context_stack else None

    def push_group(self, group_id: str) -> None:
        """Push a group onto the context stack.

        New nodes added via add_node() will become children of this group
        until pop_group() is called. Also syncs with Timeline so DSL-created
        nodes (view(), group(), etc.) are automatically linked.

        Args:
            group_id: The ID of the group to push.
        """
        self._context_stack.append(group_id)
        # Sync with Timeline for DSL node creation
        self._timeline.set_current_group(group_id)

    def pop_group(self) -> str | None:
        """Pop the current group from the context stack.

        Returns:
            The popped group ID, or None if stack was empty.
        """
        if not self._context_stack:
            return None

        popped = self._context_stack.pop()

        # Sync with Timeline: set to new top or clear
        new_top = self._context_stack[-1] if self._context_stack else None
        self._timeline.set_current_group(new_top)

        return popped

    def add_node(self, node: Any) -> str:
        """Add a node to the context graph, linking to current group if any.

        This is the primary method for adding context nodes. When a group
        is on the context stack, the node becomes a child of that group.

        Args:
            node: The ContextNode to add.

        Returns:
            The node's ID.
        """
        from activecontext.context.nodes import ContextNode

        if not isinstance(node, ContextNode):
            raise TypeError(f"Expected ContextNode, got {type(node).__name__}")

        self._timeline.context_graph.add_node(node)

        if self.current_group:
            self._timeline.context_graph.link(node.node_id, self.current_group)

        return node.node_id

    def begin_tool_use(self, tool_name: str, args: dict[str, Any] | None = None) -> str:
        """Begin a tool use scope by creating a group and pushing it to the stack.

        Creates a GroupNode for the tool call, adds a tool_call MessageNode
        as its first child, and pushes the group onto the context stack.
        Subsequent nodes added via add_node() will be children of this group.

        Args:
            tool_name: Name of the tool being invoked.
            args: Optional arguments passed to the tool.

        Returns:
            The group ID (can be used with end_tool_use).
        """
        # Create the tool group at current level
        group = GroupNode(summary_prompt=f"Tool: {tool_name}")
        self.add_node(group)  # Links to current_group if nested

        # Create tool_call message as child of the group
        tool_call = MessageNode(
            role="tool_call",
            content="",
            originator=f"tool:{tool_name}",
            tool_name=tool_name,
            tool_args=args or {},
        )
        self._timeline.context_graph.add_node(tool_call)
        self._timeline.context_graph.link(tool_call.node_id, group.node_id)

        # Push group to stack - subsequent add_node calls will be children
        self.push_group(group.node_id)

        return group.node_id

    def end_tool_use(self, summary: str | None = None) -> str | None:
        """End the current tool use scope.

        Pops the group from the context stack and optionally sets its summary.
        The summary becomes the collapsed representation of the tool use.

        Args:
            summary: Optional summary text for the group.

        Returns:
            The popped group ID, or None if no group was active.
        """
        group_id = self.pop_group()

        if group_id and summary:
            group = self._timeline.context_graph.get_node(group_id)
            if isinstance(group, GroupNode):
                group.cached_summary = summary

        return group_id

    def _add_message(self, message: Message) -> MessageNode:
        """Add a message to conversation and sync to context graph.

        Creates a MessageNode and adds it to the context graph via add_node(),
        which links to the current group if any. This enables:
        - ID-based referencing in the rendered context
        - Proper role alternation for LLM compatibility
        - Structural containment when inside a tool use scope

        Args:
            message: The Message to add

        Returns:
            The created MessageNode
        """
        # Add to conversation list
        self._message_history.append(message)

        # Create MessageNode
        msg_node = MessageNode(
            role=message.role.value,  # Convert Role enum to string
            content=message.content,
            originator=message.originator,
            tokens=500,  # Default token budget for messages
        )

        # Add via add_node() which handles current_group linking
        self.add_node(msg_node)

        return msg_node

    def save(self) -> Path:
        """Save the session to disk.

        Saves to $cwd/.ac/sessions/<session_id>.yaml

        Returns:
            Path to the saved session file.
        """
        # Get timeline statement sources
        timeline_sources = [stmt.source for stmt in self._timeline.get_statements()]

        return save_session(
            cwd=self._cwd,
            session_id=self._session_id,
            title=self._title,
            created_at=self._created_at,
            message_history=self._message_history,
            timeline_sources=timeline_sources,
            context_graph=self._timeline.context_graph,
        )

    @classmethod
    def from_file(
        cls,
        cwd: str,
        session_id: str,
        llm: LLMProvider | None = None,
        config: Config | None = None,
    ) -> Session | None:
        """Load a session from disk.

        Args:
            cwd: Project working directory.
            session_id: Session identifier.
            llm: Optional LLM provider.
            config: Optional config.

        Returns:
            Loaded Session, or None if file doesn't exist.
        """
        from activecontext.core.llm.provider import Message, Role

        session_path = get_session_path(cwd, session_id)
        data = load_session_data(session_path)
        if not data:
            return None

        # Reconstruct the context graph
        context_graph = ContextGraph.from_dict(data.context_graph)

        # Create timeline with the restored graph
        timeline = Timeline(
            session_id=session_id,
            cwd=cwd,
            context_graph=context_graph,
        )

        # Restore timeline statement history (for replay tracking)
        # Note: We don't re-execute, just record the sources
        for source in data.timeline:
            from activecontext.session.protocols import Statement
            stmt = Statement(
                statement_id=str(uuid.uuid4()),
                index=len(timeline._statements),
                source=source,
                timestamp=data.created_at.timestamp(),
            )
            timeline._statements.append(stmt)

        # Create session
        session = cls(
            session_id=session_id,
            cwd=cwd,
            timeline=timeline,
            llm=llm,
            config=config,
            title=data.title,
            created_at=data.created_at,
        )

        # Restore conversation
        for msg_data in data.conversation:
            role_str = msg_data.get("role", "user")
            role = Role.USER if role_str == "user" else Role.ASSISTANT
            msg = Message(
                role=role,
                content=msg_data.get("content", ""),
                originator=msg_data.get("originator"),
            )
            session._message_history.append(msg)

        # Link session's SessionNode to the restored graph's session node (if exists)
        restored_session_node = context_graph.get_node("session")
        if isinstance(restored_session_node, SessionNode):
            session._session_node = restored_session_node
        # If no session node was saved, one was already created in __init__

        # Link session's MCPManagerNode to the restored graph's node (if exists)
        restored_mcp_manager = context_graph.get_node("mcp_manager")
        if isinstance(restored_mcp_manager, MCPManagerNode):
            session._mcp_manager_node = restored_mcp_manager
        # If no mcp_manager node was saved, one was already created in __init__

        # Link session's user_messages group to the restored graph's node (if exists)
        restored_user_messages = context_graph.get_node("user_messages")
        if isinstance(restored_user_messages, GroupNode):
            session._user_messages_group = restored_user_messages

        # Link session's alerts group to the restored graph's node (if exists)
        restored_alerts = context_graph.get_node("alerts")
        if isinstance(restored_alerts, GroupNode):
            session._alerts_group = restored_alerts

        # Restore text buffers for virtual content (system prompts)
        session._restore_system_prompt_buffer()

        return session

    def _build_projection_engine(self, config: Config | None) -> ProjectionEngine:
        """Build ProjectionEngine from config or defaults."""
        # Budget removed - agent manages context via node visibility and line ranges
        return ProjectionEngine()

    async def prompt(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process a user prompt.

        If an LLM provider is configured:
        1. Build projection from current context
        2. Send prompt + projection to LLM
        3. Stream response, parsing and executing code blocks
        4. Run tick phase

        If no LLM:
        - Execute content directly if it looks like Python code
        - Otherwise echo the prompt
        """
        self._cancelled = False

        if self._cancelled:
            return

        if self._llm:
            # LLM-powered mode
            async for update in self._prompt_with_llm(content):
                yield update
        else:
            # Direct execution mode (fallback)
            async for update in self._prompt_direct(content):
                yield update

        # Final projection
        projection = self.get_projection()
        yield SessionUpdate(
            kind=UpdateKind.PROJECTION_READY,
            session_id=self._session_id,
            payload={
                "handles": projection.handles,
            },
            timestamp=time.time(),
        )

    def queue_user_message(self, content: str, message_id: str | None = None) -> MessageNode:
        """Queue a user message without blocking.

        Creates a MessageNode for the user's message, adds it to the user_messages
        group, and wakes the agent if idle.

        Args:
            content: The message content
            message_id: Optional explicit message ID (auto-generated if not provided)

        Returns:
            The created MessageNode
        """
        if message_id is None:
            message_id = f"msg_{uuid.uuid4().hex[:8]}"

        msg = MessageNode(
            node_id=message_id,
            role="user",
            content=content,
            originator="user",
            expansion=Expansion.DETAILS,
            mode="running",
        )

        # Add to context graph
        self._timeline.context_graph.add_node(msg)

        # Link to user_messages group if it exists
        if self._user_messages_group:
            self._timeline.context_graph.link(message_id, "user_messages")

        # Wake agent if idle
        self._wake_event.set()

        log.debug(f"Queued user message {message_id}: {content[:50]}...")
        return msg

    def has_pending_messages(self) -> bool:
        """Check if there are unprocessed user messages in the queue."""
        if not self._user_messages_group:
            return False

        children = self._timeline.context_graph.get_children("user_messages")
        for child in children:
            if isinstance(child, MessageNode):
                # Check if message has been processed (via tags)
                if not child.tags.get("processed", False):
                    return True
        return False

    def get_pending_messages(self) -> list[MessageNode]:
        """Get all unprocessed user messages."""
        if not self._user_messages_group:
            return []

        messages: list[MessageNode] = []
        children = self._timeline.context_graph.get_children("user_messages")
        for child in children:
            if isinstance(child, MessageNode) and not child.tags.get("processed", False):
                messages.append(child)
        return messages

    def mark_message_processed(self, message_id: str) -> None:
        """Mark a message as processed."""
        node = self._timeline.context_graph.get_node(message_id)
        if isinstance(node, MessageNode):
            node.tags["processed"] = True

    async def _prompt_with_llm(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process prompt using the LLM provider.

        Runs an agent loop: LLM responds, code is executed, results feed back
        to LLM until it calls done() or produces no code blocks.

        The LLM only sees the projection - no system prompt. User and assistant
        messages are added to the context graph and appear in the projection
        based on their visibility state.
        """
        import os

        from activecontext.core.llm.provider import Message, Role
        from activecontext.core.prompts import parse_response

        # Reset done signal at start of each prompt
        self._timeline.reset_done()

        max_iterations = 10  # Safety limit
        iteration = 0

        # Add initial user message to context graph (will appear in projection)
        self._add_message(
            Message(role=Role.USER, content=content, originator="user")
        )

        while iteration < max_iterations:
            iteration += 1
            turn_start = time.time()

            # Build projection (renders visible nodes from context graph)
            projection = self.get_projection()
            projection_content = projection.render()

            # Debug logging
            if os.environ.get("ACTIVECONTEXT_DEBUG"):
                tokens_est = len(projection_content) // 4 if projection_content else 0
                log.debug("=== ITERATION %d ===", iteration)
                log.debug("=== PROJECTION (%d tokens) ===", tokens_est)
                log.debug("%s", projection_content or "(empty)")
                log.debug("=== END PROJECTION ===")

            # Send only the projection to the LLM (no system prompt)
            messages = [
                Message(role=Role.USER, content=projection_content or ""),
            ]

            # Stream response from LLM
            full_response = ""
            async for chunk in self._llm.stream(messages):  # type: ignore[union-attr]
                if self._cancelled:
                    return
                if chunk.text:
                    full_response += chunk.text
                    yield SessionUpdate(
                        kind=UpdateKind.RESPONSE_CHUNK,
                        session_id=self._session_id,
                        payload={"text": chunk.text},
                        timestamp=time.time(),
                    )

            # Add assistant response to context graph and message history
            self._add_message(
                Message(role=Role.ASSISTANT, content=full_response, originator="agent")
            )

            # Parse response and execute code blocks
            parsed = parse_response(full_response)
            code_blocks = parsed.code_blocks
            execution_results: list[str] = []

            for code in code_blocks:
                if self._cancelled:
                    return
                async for update in self._execute_code(code):
                    yield update
                    # Collect execution output for feedback
                    if update.kind == UpdateKind.STATEMENT_EXECUTED:
                        stdout = update.payload.get("stdout", "")
                        stderr = update.payload.get("stderr", "")
                        exception = update.payload.get("exception")
                        if stdout:
                            execution_results.append(f"Output:\n{stdout}")
                        if stderr:
                            execution_results.append(f"Stderr:\n{stderr}")
                        if exception:
                            execution_results.append(
                                f"Error: {exception.get('type')}: {exception.get('message')}"
                            )

            # Run tick phase
            tick_updates = await self.tick()
            for update in tick_updates:
                yield update

            # Record turn statistics in SessionNode
            turn_duration_ms = (time.time() - turn_start) * 1000
            tokens_used = len(full_response) // 4  # Rough estimate
            action_desc = None
            if code_blocks:
                action_desc = f"Executed {len(code_blocks)} code block(s)"
            if self._session_node:
                self._session_node.record_turn(
                    tokens_used=tokens_used,
                    duration_ms=turn_duration_ms,
                    action_description=action_desc,
                )

            # Check if agent called done()
            if self._timeline.is_done():
                log.debug("Agent called done(), stopping loop")
                break

            # If no code was executed, the agent is done (legacy behavior)
            if not code_blocks:
                log.debug("No code blocks, stopping loop")
                break

            # Add execution results as a message (will appear in next projection)
            if execution_results:
                result_content = "Execution results:\n" + "\n".join(execution_results)
            else:
                result_content = "Code executed successfully."
            self._add_message(
                Message(role=Role.USER, content=result_content, originator="system")
            )

    async def _prompt_direct(self, content: str) -> AsyncIterator[SessionUpdate]:
        """Process prompt in direct execution mode (no LLM)."""
        # Check if content looks like Python code
        is_code = (
            any(
                content.strip().startswith(prefix)
                for prefix in ("import ", "from ", "def ", "class ", "=", "view(", "group(")
            )
            or "=" in content
        )

        if is_code:
            async for update in self._execute_code(content):
                yield update
            tick_updates = await self.tick()
            for update in tick_updates:
                yield update
        else:
            yield SessionUpdate(
                kind=UpdateKind.RESPONSE_CHUNK,
                session_id=self._session_id,
                payload={"text": f"[No LLM configured] Received: {content}"},
                timestamp=time.time(),
            )

    async def _execute_code(self, source: str) -> AsyncIterator[SessionUpdate]:
        """Execute a code block and yield updates."""
        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_PARSED,
            session_id=self._session_id,
            payload={"source": source},
            timestamp=time.time(),
        )

        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_EXECUTING,
            session_id=self._session_id,
            payload={"source": source},
            timestamp=time.time(),
        )

        result = await self._timeline.execute_statement(source)

        yield SessionUpdate(
            kind=UpdateKind.STATEMENT_EXECUTED,
            session_id=self._session_id,
            payload={
                "execution_id": result.execution_id,
                "statement_id": result.statement_id,
                "status": result.status.value,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exception": result.exception,
                "state_trace": {
                    "added": result.state_trace.added,
                    "changed": result.state_trace.changed,
                    "deleted": result.state_trace.deleted,
                },
                "duration_ms": result.duration_ms,
            },
            timestamp=time.time(),
        )

    async def tick(self) -> list[SessionUpdate]:
        """Run the tick phase for context graph nodes.

        Tick ordering:
        1. Process pending shell results (async completions from background tasks)
        2. Turn ticks for running nodes (calls Recompute → notify_parents cascade)
        3. Periodic ticks based on tick_frequency
        4. Group summaries auto-invalidated via on_child_changed() during cascade
        5. Check wait conditions and prepare wake prompts
        """
        updates: list[SessionUpdate] = []
        timestamp = time.time()

        # 1. Process pending shell results from background tasks
        # This applies async results and triggers node change notifications
        updated_shell_nodes = self._timeline.process_pending_shell_results()
        for node_id in updated_shell_nodes:
            node = self._timeline.context_graph.get_node(node_id)
            if node:
                updates.append(
                    SessionUpdate(
                        kind=UpdateKind.NODE_CHANGED,
                        session_id=self._session_id,
                        payload={
                            "node_id": node_id,
                            "node_type": node.node_type,
                            "change": "shell_completed",
                            "digest": node.GetDigest(),
                        },
                        timestamp=timestamp,
                    )
                )

        # 2. Process pending file changes from file watcher
        wake_prompts = self._timeline.process_file_changes()
        if wake_prompts:
            # Wake the agent loop if any WAKE events fired
            self._wake_event.set()

        # 3-5. Process running nodes with tick frequencies
        context_graph = self._timeline.context_graph
        running_nodes = context_graph.get_running_nodes()

        for node in running_nodes:
            tick_kind: str | None = None

            # Check tick frequency
            if node.tick_frequency is None:
                # No tick frequency set, skip
                continue

            if node.tick_frequency.mode == "turn":
                # Turn tick: recompute on every tick (replaces "Sync")
                tick_kind = "turn"
            elif node.tick_frequency.mode == "periodic":
                # Periodic tick: check interval
                if node.tick_frequency.interval is None:
                    log.warning("Periodic tick frequency without interval for node %s", node.node_id)
                    continue

                if timestamp - node.updated_at >= node.tick_frequency.interval:
                    tick_kind = "periodic"
            elif node.tick_frequency.mode == "async":
                # Async mode - not yet implemented
                pass
            elif node.tick_frequency.mode == "never":
                # Never tick
                continue

            if tick_kind:
                # Call Recompute which triggers notify_parents() cascade
                node.Recompute()
                node.updated_at = timestamp

                updates.append(
                    SessionUpdate(
                        kind=UpdateKind.TICK_APPLIED,
                        session_id=self._session_id,
                        payload={
                            "node_id": node.node_id,
                            "tick_kind": tick_kind,
                            "digest": node.GetDigest(),
                        },
                        timestamp=timestamp,
                    )
                )

        # 6. Process notifications from nodes with HOLD/WAKE levels
        # Check for WAKE notifications before flush (flush clears the flag)
        should_wake = context_graph.has_wake_notification()
        notifications = context_graph.flush_notifications()
        if notifications:
            self._update_alerts_group(notifications)
            updates.append(
                SessionUpdate(
                    kind=UpdateKind.NODE_CHANGED,
                    session_id=self._session_id,
                    payload={
                        "node_id": "alerts",
                        "change": "notifications_updated",
                        "count": len(notifications),
                    },
                    timestamp=timestamp,
                )
            )

        if should_wake:
            self._wake_event.set()

        return updates

    async def cancel(self) -> None:
        """Cancel the current operation and all running shell commands."""
        self._cancelled = True
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        # Also cancel all running shell tasks
        self._timeline._shell_manager.cancel_all()

    async def run_agent_loop(self) -> AsyncIterator[SessionUpdate]:
        """Event-driven agent loop. Idle until wake, process until queue empty.

        This is the main processing loop for the async prompt model. It:
        1. Waits for a wake signal (message queued, file changed, etc.)
        2. Processes all pending messages
        3. Runs tick phase to handle async completions
        4. Yields SessionUpdate for each significant event

        Yields:
            SessionUpdate objects for streaming to the transport
        """
        self._running = True
        log.info("Agent loop started for session %s", self._session_id)

        while self._running:
            # Wait for wake signal
            try:
                await self._wake_event.wait()
            except asyncio.CancelledError:
                log.info("Agent loop cancelled for session %s", self._session_id)
                break

            self._wake_event.clear()
            log.debug("Agent loop woke for session %s", self._session_id)

            # Process all pending work
            while self._has_pending_work() and self._running:
                # Process next message
                async for update in self._process_next_message():
                    yield update

                # Run tick phase
                tick_updates = await self.tick()
                for update in tick_updates:
                    yield update

        log.info("Agent loop stopped for session %s", self._session_id)

    def _has_pending_work(self) -> bool:
        """Check if there's work to do."""
        return (
            self.has_pending_messages()
            or self._timeline.has_pending_wake_prompt()
            or len(self._timeline.get_queued_events()) > 0
        )

    async def _process_next_message(self) -> AsyncIterator[SessionUpdate]:
        """Process the next pending user message.

        Gets the oldest unprocessed message, processes it through the LLM,
        and marks it as processed.

        Yields:
            SessionUpdate objects for the message processing
        """
        messages = self.get_pending_messages()
        if not messages:
            return

        # Process oldest message first (FIFO)
        msg = messages[0]
        content = msg.content
        log.debug("Processing message %s: %s", msg.node_id, content[:50])

        # Mark as in-progress
        msg.tags["processing"] = True

        try:
            if self._llm:
                # LLM-powered mode
                async for update in self._prompt_with_llm(content):
                    yield update
            else:
                # Direct execution mode (fallback)
                async for update in self._prompt_direct(content):
                    yield update

            # Mark as processed
            self.mark_message_processed(msg.node_id)
            msg.tags.pop("processing", None)

            # Send completion notification
            yield SessionUpdate(
                kind=UpdateKind.PROJECTION_READY,
                session_id=self._session_id,
                payload={
                    "message_id": msg.node_id,
                    "completed": True,
                    "handles": self.get_projection().handles,
                },
                timestamp=time.time(),
            )

        except asyncio.CancelledError:
            msg.tags.pop("processing", None)
            raise
        except Exception as e:
            log.error("Error processing message %s: %s", msg.node_id, e)
            msg.tags.pop("processing", None)
            msg.tags["error"] = str(e)
            yield SessionUpdate(
                kind=UpdateKind.ERROR,
                session_id=self._session_id,
                payload={
                    "message_id": msg.node_id,
                    "error": str(e),
                },
                timestamp=time.time(),
            )

    def stop_agent_loop(self) -> None:
        """Stop the agent loop gracefully."""
        self._running = False
        self._wake_event.set()  # Wake it so it can exit

    def wake(self) -> None:
        """Wake the agent loop to process pending work."""
        self._wake_event.set()

    def _update_alerts_group(self, notifications: list[Notification]) -> None:
        """Update Alerts group with new notifications.

        Args:
            notifications: List of Notification objects to display
        """
        from activecontext.context.nodes import ArtifactNode

        if not self._alerts_group:
            return

        context_graph = self._timeline.context_graph

        # Clear old alert nodes
        for child in context_graph.get_children(self._alerts_group.node_id):
            context_graph.remove_node(child.node_id)

        # Add new alert nodes (as ArtifactNodes)
        for i, notif in enumerate(notifications):
            alert_node = ArtifactNode(
                node_id=f"alert_{i}",
                content=notif.header,
                artifact_type="notification",
                expansion=Expansion.DETAILS,
                tags={"level": notif.level, "source": notif.node_id},
            )
            context_graph.add_node(alert_node)
            context_graph.link(alert_node.node_id, self._alerts_group.node_id)

        # Update group visibility based on content
        self._alerts_group.expansion = Expansion.DETAILS if notifications else Expansion.HIDDEN

    def get_projection(self) -> Projection:
        """Build the LLM projection from current session state.

        Renders visible nodes from the context graph. The agent controls
        visibility by showing, hiding, expanding, and collapsing nodes.
        """
        return self._projection_engine.build(
            context_graph=self._timeline.context_graph,
            cwd=self._cwd,
            text_buffers=self._text_buffers,
        )

    def clear_message_history(self) -> None:
        """Clear the conversation history."""
        self._message_history.clear()

    def get_context_objects(self) -> dict[str, Any]:
        """Get all context objects from the graph."""
        return self._timeline.get_context_objects()

    def get_context_graph(self) -> ContextGraph:
        """Get the context graph (DAG of context nodes)."""
        return self._timeline.context_graph

    async def _setup_initial_context(self) -> None:
        """Set up initial context for a new session.

        Three-tier startup execution:
        1. Base statements: user's `statements` if set, else PACKAGE_DEFAULT_STARTUP
        2. Additional statements: user's `additional` always executes last
        3. Context guide: loaded unless skip_default_context is True

        Only runs on NEW session creation, not when loading from file.
        """
        from activecontext.config.schema import PACKAGE_DEFAULT_STARTUP, StartupConfig

        startup_config = (
            self._config.session.startup
            if self._config and self._config.session
            else StartupConfig()
        )

        # Determine base statements: project override or package defaults
        base_statements = (
            startup_config.statements
            if startup_config.statements
            else PACKAGE_DEFAULT_STARTUP
        )

        # Execute base statements
        for statement in base_statements:
            try:
                await self._timeline.execute_statement(statement)
            except Exception as e:
                log.warning(f"Startup statement failed: {statement!r}: {e}")

        # Execute user additions last (always additive)
        for statement in startup_config.additional:
            try:
                await self._timeline.execute_statement(statement)
            except Exception as e:
                log.warning(f"Additional startup statement failed: {statement!r}: {e}")

        # Load context guide (project-specific or bundled) unless skipped
        if not startup_config.skip_default_context:
            await self._load_context_guide()

        # Create metadata nodes after prompts (document order: prompt, guide, session, mcp)
        self._create_metadata_nodes()

        # Auto-connect to configured MCP servers
        await self._setup_mcp_autoconnect()



    async def _load_context_guide(self) -> None:
        """Load the context guide as a MarkdownNode if it exists.

        Checks for CONTEXT_GUIDE.md in cwd, then prompts directory.
        """
        from pathlib import Path

        guide_paths = [
            Path(self._cwd) / "CONTEXT_GUIDE.md",
            Path(__file__).parent.parent / "prompts" / "context_guide.md",
        ]

        for guide_path in guide_paths:
            if guide_path.exists():
                rel_path = guide_path.name
                try:
                    rel_path = str(guide_path.relative_to(self._cwd))
                except ValueError:
                    rel_path = str(guide_path)

                # Use forward slashes for cross-platform compatibility
                safe_path = rel_path.replace("\\", "/")
                source = f'guide = markdown("{safe_path}", tokens=1500, expansion=Expansion.DETAILS)'
                await self._timeline.execute_statement(source)
                break

    async def _setup_mcp_autoconnect(self) -> None:
        """Connect to MCP servers based on their connect mode.

        - CRITICAL: Must connect, raise error if fails
        - AUTO: Try on startup, warn if fails
        - MANUAL: Skip (user connects manually)
        - NEVER: Skip (disabled)
        """
        if not self._config or not self._config.mcp:
            return

        import logging

        from activecontext.config.schema import MCPConnectMode

        _log = logging.getLogger("activecontext.session")

        for server_config in self._config.mcp.servers:
            if server_config.connect == MCPConnectMode.NEVER:
                continue

            if server_config.connect in (MCPConnectMode.CRITICAL, MCPConnectMode.AUTO):
                try:
                    source = f'mcp_connect("{server_config.name}")'
                    await self._timeline.execute_statement(source)
                    _log.info(f"Auto-connected to MCP server '{server_config.name}'")
                except Exception as e:
                    if server_config.connect == MCPConnectMode.CRITICAL:
                        raise RuntimeError(
                            f"Critical MCP server '{server_config.name}' failed to connect: {e}"
                        ) from e
                    else:
                        _log.warning(
                            f"Failed to auto-connect to MCP server '{server_config.name}': {e}"
                        )


class SessionManager:
    """Manages multiple sessions with 1:1 session-timeline mapping.

    Implements SessionManagerProtocol.
    """

    def __init__(self, default_llm: LLMProvider | None = None) -> None:
        """Initialize the session manager.

        Args:
            default_llm: Default LLM provider for new sessions
        """
        self._sessions: dict[str, Session] = {}
        self._default_llm = default_llm
        # Track config reload unregister callbacks per session
        self._reload_unregisters: dict[str, Callable[[], None]] = {}

    def set_default_llm(self, llm: LLMProvider | None) -> None:
        """Set the default LLM provider for new sessions."""
        self._default_llm = llm

    async def create_session(
        self,
        cwd: str,
        session_id: str | None = None,
        llm: LLMProvider | None = None,
        terminal_executor: TerminalExecutor | None = None,
        permission_requester: PermissionRequester | None = None,
        shell_permission_requester: ShellPermissionRequester | None = None,
        website_permission_requester: WebsitePermissionRequester | None = None,
        import_permission_requester: ImportPermissionRequester | None = None,
    ) -> Session:
        """Create a new session with its own timeline.

        Args:
            cwd: Working directory for the session
            session_id: Optional specific ID; generated if not provided
            llm: Optional LLM provider; uses default if not provided
            terminal_executor: Optional terminal executor for shell commands;
                defaults to SubprocessTerminalExecutor if not provided
            permission_requester: Optional callback for ACP file permission prompts;
                async (session_id, path, mode) -> (granted, persist)
            shell_permission_requester: Optional callback for ACP shell permission prompts;
                async (session_id, command, args) -> (granted, persist)
            website_permission_requester: Optional callback for ACP website permission prompts;
                async (session_id, url, method) -> (granted, persist)
            import_permission_requester: Optional callback for ACP import permission prompts;
                async (session_id, module) -> (granted, persist, include_submodules)
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id in self._sessions:
            raise ValueError(f"Session {session_id} already exists")

        # Load project-level config (merges with system/user config)
        project_config = self._load_project_config(cwd)

        # Create permission manager from config
        sandbox_config = project_config.sandbox if project_config else None
        permission_manager = PermissionManager.from_config(cwd, sandbox_config)

        # Create shell permission manager from config
        shell_permission_manager = ShellPermissionManager.from_config(sandbox_config)

        # Create website permission manager from config
        from activecontext.session.permissions import WebsitePermissionManager

        website_permission_manager = WebsitePermissionManager.from_config(
            sandbox_config, Path(cwd)
        )

        # Create import guard from config
        import_config = sandbox_config.imports if sandbox_config else None
        import_guard = ImportGuard.from_config(import_config)

        # Create scratchpad manager for work coordination
        from activecontext.coordination import ScratchpadManager

        scratchpad_manager = ScratchpadManager(cwd)

        # Get MCP config if available
        mcp_config = project_config.mcp if project_config else None

        # Create context graph - Session owns this, Timeline uses it
        context_graph = ContextGraph()

        timeline = Timeline(
            session_id,
            context_graph=context_graph,
            cwd=cwd,
            permission_manager=permission_manager,
            terminal_executor=terminal_executor,
            permission_requester=permission_requester,  # type: ignore[arg-type]
            import_guard=import_guard,
            import_permission_requester=import_permission_requester,  # type: ignore[arg-type]
            shell_permission_manager=shell_permission_manager,
            shell_permission_requester=shell_permission_requester,  # type: ignore[arg-type]
            website_permission_manager=website_permission_manager,
            website_permission_requester=website_permission_requester,  # type: ignore[arg-type]
            scratchpad_manager=scratchpad_manager,
            mcp_config=mcp_config,
        )
        session = Session(
            session_id=session_id,
            cwd=cwd,
            timeline=timeline,
            llm=llm or self._default_llm,
            config=project_config,
        )
        self._sessions[session_id] = session

        # Register for config reload to update permissions
        unregister = self._setup_permission_reload(
            session_id, cwd, permission_manager, shell_permission_manager, website_permission_manager, import_guard
        )
        self._reload_unregisters[session_id] = unregister

        # Initialize with example context view if guide exists
        await session._setup_initial_context()

        return session

    def _setup_permission_reload(
        self,
        session_id: str,
        cwd: str,
        permission_manager: PermissionManager,
        shell_permission_manager: ShellPermissionManager | None = None,
        website_permission_manager: Any | None = None,
        import_guard: ImportGuard | None = None,
    ) -> Callable[[], None]:
        """Set up config reload callback for a session's permission managers.

        Args:
            session_id: Session ID for tracking.
            cwd: Working directory for config loading.
            permission_manager: PermissionManager to update on reload.
            shell_permission_manager: Optional ShellPermissionManager to update on reload.
            website_permission_manager: Optional WebsitePermissionManager to update on reload.
            import_guard: Optional ImportGuard to update on reload.

        Returns:
            Unregister function to remove the callback.
        """
        try:
            from activecontext.config.loader import on_config_reload

            def on_reload(new_config: Config) -> None:
                # Only update if session still exists
                if session_id in self._sessions:
                    sandbox_config = new_config.sandbox if new_config else None
                    permission_manager.reload(sandbox_config)
                    log.debug(
                        "Permissions reloaded for session %s: %d rules",
                        session_id,
                        len(permission_manager.rules),
                    )
                    if shell_permission_manager:
                        shell_permission_manager.reload(sandbox_config)
                        log.debug(
                            "Shell permissions reloaded for session %s: %d rules",
                            session_id,
                            len(shell_permission_manager.rules),
                        )
                    if website_permission_manager:
                        website_permission_manager.reload(sandbox_config)
                        log.debug(
                            "Website permissions reloaded for session %s: %d rules",
                            session_id,
                            len(website_permission_manager.rules),
                        )
                    if import_guard:
                        import_config = sandbox_config.imports if sandbox_config else None
                        import_guard.reload(import_config)
                        log.debug(
                            "Import whitelist reloaded for session %s: %d modules",
                            session_id,
                            len(import_guard.allowed_modules),
                        )

            return on_config_reload(on_reload)
        except ImportError:
            log.debug("Config reload not available")
            return lambda: None

    def _load_project_config(self, cwd: str) -> Config | None:
        """Load project-level config for a session.

        Args:
            cwd: Working directory (project root) for the session.

        Returns:
            Merged config including project-level settings, or None if unavailable.
        """
        try:
            from activecontext.config import load_config

            return load_config(session_root=cwd)
        except ImportError:
            log.debug("Config module not available")
            return None
        except Exception as e:
            log.warning("Failed to load project config: %s", e)
            return None

    async def get_session(self, session_id: str) -> Session | None:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    async def load_session(
        self,
        session_id: str,
        cwd: str,
    ) -> Session | None:
        """Load a previously persisted session.

        TODO: Implement persistence loading. For now, returns None.
        """
        return None

    async def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())

    async def close_session(self, session_id: str) -> None:
        """Close and clean up a session."""
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)

            # Unregister from work coordination scratchpad
            if session._timeline._scratchpad_manager:
                session._timeline._scratchpad_manager.unregister()

            await session.cancel()

            # Unregister config reload callback
            if session_id in self._reload_unregisters:
                unregister = self._reload_unregisters.pop(session_id)
                unregister()
