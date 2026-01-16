"""Entry point for running ActiveContext as an ACP agent.

Usage:
    python -m activecontext

    # For testing with a file:
    cat tests/fixtures/init_and_session.jsonl | python -m activecontext

This starts the ACP agent listening on stdin/stdout for JSON-RPC
messages from an ACP client (Rider, Zed, etc.).
"""

import os
import sys
import threading


_LOG_FILE = None


def _log(msg: str) -> None:
    """Log for debugging - to file if ACTIVECONTEXT_LOG is set, else stderr."""
    global _LOG_FILE
    import time

    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}\n"

    # Initialize log file on first call
    if _LOG_FILE is None:
        log_path = os.environ.get("ACTIVECONTEXT_LOG")
        if log_path:
            try:
                _LOG_FILE = open(log_path, "a", encoding="utf-8")
            except Exception:
                _LOG_FILE = False  # type: ignore[assignment]
        else:
            _LOG_FILE = False  # type: ignore[assignment]

    if _LOG_FILE:
        _LOG_FILE.write(line)
        _LOG_FILE.flush()
    else:
        print(f"[activecontext] {msg}", file=sys.stderr, flush=True)


def _setup_parent_death_monitor() -> None:
    """Exit when parent process dies.

    This prevents orphan processes when the IDE crashes or is killed.
    """
    parent_pid = os.getppid()

    # On Linux, use prctl to get SIGTERM when parent dies
    if sys.platform == "linux":
        try:
            import ctypes
            import signal

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
            _log(f"Registered for parent death signal (parent={parent_pid})")
            return
        except Exception:
            pass  # Fall through to polling

    # Cross-platform fallback: poll parent PID and check stdio
    def monitor() -> None:
        import time

        while True:
            time.sleep(2)

            # Check if stdin/stdout are still open
            try:
                if sys.stdin.closed or sys.stdout.closed:
                    _log("stdio closed, exiting")
                    os._exit(0)
            except Exception:
                _log("stdio check failed, exiting")
                os._exit(0)

            # Check if parent still exists
            try:
                if sys.platform == "win32":
                    # On Windows, check if process exists
                    import ctypes

                    kernel32 = ctypes.windll.kernel32
                    SYNCHRONIZE = 0x00100000
                    handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
                    if handle:
                        kernel32.CloseHandle(handle)
                    else:
                        _log(f"Parent process {parent_pid} died, exiting")
                        os._exit(0)
                else:
                    # On Unix, send signal 0 to check existence
                    os.kill(parent_pid, 0)
            except (OSError, ProcessLookupError):
                _log(f"Parent process {parent_pid} died, exiting")
                os._exit(0)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    _log(f"Monitoring parent process {parent_pid}")


async def _main() -> None:
    """Async entry point with proper cleanup."""
    import asyncio

    import acp
    from acp.agent.connection import AgentSideConnection
    from acp.connection import StreamDirection, StreamEvent
    from acp.stdio import stdio_streams

    from activecontext.transport.acp.agent import create_agent

    _log("Creating agent...")
    agent = create_agent()

    _log(f"Agent ready, model={agent._current_model_id}")

    # Track activity for inactivity timeout
    last_activity = [asyncio.get_event_loop().time()]
    INACTIVITY_TIMEOUT = 300  # 5 minutes

    def log_message(event: StreamEvent) -> None:
        """Log all ACP messages for debugging."""
        direction = "<<" if event.direction == StreamDirection.INCOMING else ">>"
        method = event.message.get("method", "response")
        msg_id = event.message.get("id", "-")
        _log(f"{direction} {method} (id={msg_id})")
        last_activity[0] = asyncio.get_event_loop().time()

    async def inactivity_monitor() -> None:
        """Exit if no activity for too long."""
        while True:
            await asyncio.sleep(30)
            elapsed = asyncio.get_event_loop().time() - last_activity[0]
            if elapsed > INACTIVITY_TIMEOUT:
                _log(f"No activity for {elapsed:.0f}s, exiting...")
                os._exit(0)

    # Create connection manually so we can clean up properly
    output_stream, input_stream = await stdio_streams()
    conn = AgentSideConnection(
        agent,
        input_stream,
        output_stream,
        listening=False,
        use_unstable_protocol=True,
    )

    # Add message observer for logging
    conn._conn.add_observer(log_message)

    # Start inactivity monitor
    monitor_task = asyncio.create_task(inactivity_monitor())

    try:
        await conn.listen()
    finally:
        # Ensure connection is properly closed
        _log("Connection closed, cleaning up...")
        monitor_task.cancel()
        await conn.close()


def main() -> None:
    """Run the ActiveContext ACP agent."""
    import asyncio

    _log("Starting...")

    # Ensure we exit when parent dies
    _setup_parent_death_monitor()

    try:
        asyncio.run(_main())
    finally:
        # Ensure process exits even if there are lingering resources
        _log("Exiting...")
        os._exit(0)


if __name__ == "__main__":
    main()
