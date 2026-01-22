"""Shell command execution manager.

Manages async shell execution, background tasks, and permission checks.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from activecontext.context.nodes import ShellNode, ShellStatus
from activecontext.context.state import Expansion
from activecontext.terminal.result import ShellResult

if TYPE_CHECKING:
    from activecontext.context.graph import ContextGraph
    from activecontext.session.permissions import ShellPermissionManager
    from activecontext.terminal.protocol import TerminalExecutor

    # Type alias for shell permission requester callback
    ShellPermissionRequester = Any  # Callable[[str, str, list[str] | None], Awaitable[tuple[bool, bool]]]


class ShellManager:
    """Manages shell command execution and background task tracking.

    Responsibilities:
    - Execute shell commands asynchronously
    - Handle permission checks before execution
    - Track running tasks and pending results
    - Process results at tick time
    - Cancel running/pending tasks
    """

    def __init__(
        self,
        *,
        context_graph: ContextGraph,
        terminal_executor: TerminalExecutor,
        shell_permission_manager: ShellPermissionManager | None = None,
        shell_permission_requester: ShellPermissionRequester | None = None,
        session_id: str,
        cwd: str,
    ):
        """Initialize shell manager.

        Args:
            context_graph: The session's context graph for adding ShellNodes
            terminal_executor: Executor for running shell commands
            shell_permission_manager: Optional permission manager for sandbox control
            shell_permission_requester: Optional callback to request permissions from user
            session_id: Session ID for permission requests
            cwd: Default working directory for shell commands
        """
        self._context_graph = context_graph
        self._terminal_executor = terminal_executor
        self._shell_permission_manager = shell_permission_manager
        self._shell_permission_requester = shell_permission_requester
        self._session_id = session_id
        self._cwd = cwd

        # Pending results from background tasks (processed at tick)
        self._pending_shell_results: list[tuple[str, ShellResult]] = []

        # Running background tasks keyed by node_id
        self._shell_tasks: dict[str, asyncio.Task[None]] = {}

    def execute(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = 30.0,
        *,
        tokens: int = 2000,
        state: Expansion = Expansion.DETAILS,
    ) -> ShellNode:
        """Execute a shell command asynchronously, returning a ShellNode.

        Creates a ShellNode in the context graph and starts the subprocess
        in the background. The node's status updates when the command completes,
        and change notifications propagate up the DAG at tick time.

        Args:
            command: The command to execute (e.g., "pytest", "git").
            args: Optional list of arguments (e.g., ["tests/", "-v"]).
            cwd: Working directory. If None, uses session cwd.
            env: Additional environment variables.
            timeout: Timeout in seconds (default: 30). None for no timeout.
            tokens: Token budget for rendering output (default: 2000).
            state: Initial rendering state (default: DETAILS).

        Returns:
            ShellNode that tracks the command execution.
        """
        # Create the ShellNode
        node = ShellNode(
            command=command,
            args=args or [],
            tokens=tokens,
            state=state,
        )

        # Add to context graph
        self._context_graph.add_node(node)

        # Start background execution
        task = asyncio.create_task(
            self._background_task(
                node_id=node.node_id,
                command=command,
                args=args,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )
        )
        self._shell_tasks[node.node_id] = task

        return node

    async def _background_task(
        self,
        node_id: str,
        command: str,
        args: list[str] | None,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout: float | None,
    ) -> None:
        """Background task that executes a shell command and stores the result.

        This runs in the background while the agent continues. When complete,
        it stores the result in _pending_shell_results for processing at tick.
        """
        # Get the node (stays PENDING until permission is granted)
        node = self._context_graph.get_node(node_id)

        result: ShellResult

        # Check permission if manager is configured
        if self._shell_permission_manager:
            if not self._shell_permission_manager.check_access(command, args):
                # Permission denied - try to request
                if self._shell_permission_requester:
                    try:
                        granted, persist = await self._shell_permission_requester(
                            self._session_id, command, args
                        )
                    except asyncio.CancelledError:
                        # Cancelled during permission request
                        full_cmd = f"{command} {' '.join(args or [])}"
                        result = ShellResult(
                            command=full_cmd,
                            exit_code=-1,
                            output="Cancelled",
                            truncated=False,
                            status="cancelled",
                            signal="SIGTERM",
                            duration_ms=0,
                        )
                        self._pending_shell_results.append((node_id, result))
                        return

                    if granted:
                        if persist:
                            # "Allow always" - write to config file
                            from activecontext.session.permissions import (
                                write_shell_permission_to_config,
                            )

                            write_shell_permission_to_config(
                                Path(self._cwd), command, args
                            )
                            # Reload config to pick up new rule
                            from activecontext.config import load_config

                            config = load_config(session_root=self._cwd)
                            self._shell_permission_manager.reload(config.sandbox)
                        else:
                            # "Allow once" - grant temporary access
                            self._shell_permission_manager.grant_temporary(command, args)
                    else:
                        # Denied - store error result
                        full_cmd = f"{command} {' '.join(args or [])}"
                        result = ShellResult(
                            command=full_cmd,
                            exit_code=126,  # Permission denied exit code
                            output=f"Shell command denied by sandbox policy: {full_cmd}",
                            truncated=False,
                            status="error",
                            signal=None,
                            duration_ms=0,
                        )
                        self._pending_shell_results.append((node_id, result))
                        return
                else:
                    # No requester available - store error result
                    full_cmd = f"{command} {' '.join(args or [])}"
                    result = ShellResult(
                        command=full_cmd,
                        exit_code=126,
                        output=f"Shell command denied by sandbox policy: {full_cmd}",
                        truncated=False,
                        status="error",
                        signal=None,
                        duration_ms=0,
                    )
                    self._pending_shell_results.append((node_id, result))
                    return

        # Permission granted (or no manager) - mark as running and execute
        if isinstance(node, ShellNode):
            node.shell_status = ShellStatus.RUNNING
            node.started_at_exec = time.time()

        try:
            result = await self._terminal_executor.execute(
                command=command,
                args=args,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )
        except asyncio.CancelledError:
            full_cmd = f"{command} {' '.join(args or [])}"
            result = ShellResult(
                command=full_cmd,
                exit_code=-1,
                output="Cancelled",
                truncated=False,
                status="cancelled",
                signal="SIGTERM",
                duration_ms=0,
            )
        except Exception as e:
            full_cmd = f"{command} {' '.join(args or [])}"
            result = ShellResult(
                command=full_cmd,
                exit_code=-1,
                output=f"Error: {e}",
                truncated=False,
                status="error",
                signal=None,
                duration_ms=0,
            )

        # Store result for processing at tick time
        self._pending_shell_results.append((node_id, result))

    def process_pending_results(self) -> list[str]:
        """Process pending shell results and update nodes.

        Called during tick to apply async shell results. This triggers
        node change notifications that propagate up the DAG.

        Returns:
            List of node IDs that were updated.
        """
        updated_nodes: list[str] = []

        while self._pending_shell_results:
            node_id, result = self._pending_shell_results.pop(0)

            node = self._context_graph.get_node(node_id)
            if not isinstance(node, ShellNode):
                continue

            # Apply the result - this triggers _mark_changed() and notifications
            if result.status == "timeout":
                node.set_timeout(result.output, result.duration_ms)
            elif result.signal:
                node.set_completed(
                    exit_code=result.exit_code or -1,
                    output=result.output,
                    duration_ms=result.duration_ms,
                    truncated=result.truncated,
                    signal=result.signal,
                )
            else:
                node.set_completed(
                    exit_code=result.exit_code or 0,
                    output=result.output,
                    duration_ms=result.duration_ms,
                    truncated=result.truncated,
                )

            updated_nodes.append(node_id)

            # Clean up task reference
            self._shell_tasks.pop(node_id, None)

        return updated_nodes

    def cancel(self, node_id: str) -> bool:
        """Cancel a running shell command.

        Args:
            node_id: The ShellNode ID to cancel.

        Returns:
            True if a task was cancelled, False if not found/already done.
        """
        task = self._shell_tasks.get(node_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    def cancel_all(self) -> int:
        """Cancel all running shell commands.

        Returns:
            Number of tasks cancelled.
        """
        cancelled = 0
        for node_id, task in list(self._shell_tasks.items()):
            if not task.done():
                task.cancel()
                cancelled += 1
        return cancelled

    def has_pending_tasks(self) -> bool:
        """Check if there are any running shell tasks.

        Returns:
            True if any shell tasks are still running.
        """
        return len(self._shell_tasks) > 0
