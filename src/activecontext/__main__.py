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


def _log(msg: str) -> None:
    """Log to stderr for debugging."""
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


def main() -> None:
    """Run the ActiveContext ACP agent."""
    import asyncio

    _log("Starting...")

    # Ensure we exit when parent dies
    _setup_parent_death_monitor()

    import acp

    _log("Importing agent...")
    from activecontext.transport.acp.agent import create_agent

    _log("Creating agent...")
    agent = create_agent()

    _log(f"Agent ready, model={agent._current_model_id}")
    asyncio.run(acp.run_agent(agent, use_unstable_protocol=True))


if __name__ == "__main__":
    main()
