"""Entry point for running ActiveContext as an ACP agent.

Usage:
    python -m activecontext

    # For testing with a file:
    cat tests/fixtures/init_and_session.jsonl | python -m activecontext

This starts the ACP agent listening on stdin/stdout for JSON-RPC
messages from an ACP client (Rider, Zed, etc.).
"""

import os

from activecontext.logging import get_logger, setup_logging

log = get_logger()


def _expand_env_vars() -> None:
    """Expand ${VAR_NAME} references in environment variables.
    
    This allows acp.json to use:
        "env": { "OPENAI_API_KEY": "${OPENAI_API_KEY}" }
    
    To pull from the system environment.
    """
    import re
    
    pattern = re.compile(r'\$\{([^}]+)\}')
    
    # Iterate over a copy since we're modifying os.environ
    for key, value in list(os.environ.items()):
        if not isinstance(value, str):
            continue
            
        # Find all ${VAR_NAME} patterns
        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            # Get from original environment (not the modified one)
            return os.environ.get(var_name, match.group(0))
        
        expanded = pattern.sub(replace_var, value)
        if expanded != value:
            os.environ[key] = expanded
            log.debug("Expanded env var: %s", key)


def _setup_parent_death_monitor() -> None:
    """Exit when parent process dies.

    This prevents orphan processes when the IDE crashes or is killed.
    On Windows with venvs, the python.exe is a launcher that spawns the real
    Python interpreter. When the launcher dies, we need to detect this and exit.
    """
    import sys
    import threading

    if sys.platform != "win32":
        # On Unix, we could use prctl(PR_SET_PDEATHSIG) but it's complex
        # Rely on stdio handle monitoring instead
        return

    parent_pid = os.getppid()
    log.debug("Parent PID: %d, monitoring for exit", parent_pid)

    def monitor_parent() -> None:
        """Background thread that exits when parent dies."""
        import time

        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            INFINITE = 0xFFFFFFFF

            # Open handle to parent process
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
            if not handle:
                log.warning("Could not open parent process handle, falling back to polling")
                # Fall back to polling
                while True:
                    time.sleep(1.0)
                    if os.getppid() != parent_pid:
                        log.info("Parent process changed, exiting")
                        os._exit(0)
                return

            try:
                # Wait for parent to exit
                result = kernel32.WaitForSingleObject(handle, INFINITE)
                log.info("Parent process exited (wait result=%d), terminating", result)
                os._exit(0)
            finally:
                kernel32.CloseHandle(handle)

        except Exception as e:
            log.warning("Parent monitor error: %s, falling back to polling", e)
            # Fall back to polling
            while True:
                time.sleep(1.0)
                try:
                    if os.getppid() != parent_pid:
                        log.info("Parent process changed, exiting")
                        os._exit(0)
                except Exception:
                    os._exit(0)

    # Start monitor thread
    thread = threading.Thread(target=monitor_parent, daemon=True)
    thread.start()
    log.debug("Parent death monitor started")


async def _main() -> None:
    """Async entry point with proper cleanup."""
    import asyncio
    import json

    from acp.agent.connection import AgentSideConnection
    from acp.connection import StreamDirection, StreamEvent
    from acp.stdio import stdio_streams

    from activecontext.transport.acp.agent import create_agent

    log.info("Creating agent...")
    agent = create_agent()

    if agent._current_model_id:
        log.info("Agent ready, model=%s", agent._current_model_id)
    else:
        log.warning(
            "Agent ready, no LLM configured "
            "(set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or DEEPSEEK_API_KEY)"
        )

    def log_message(event: StreamEvent) -> None:
        """Log all ACP messages for debugging."""
        direction = "<<" if event.direction == StreamDirection.INCOMING else ">>"
        method = event.message.get("method", "response")
        msg_id = event.message.get("id", "-")
        msg_str = json.dumps(event.message, default=str)
        msg_len = len(msg_str)

        if method == "response":
            # Log response details including stop_reason for debugging
            # Note: ACP uses camelCase "stopReason" in JSON
            result = event.message.get("result", {})
            stop_reason = (
                result.get("stopReason", "unknown")
                if isinstance(result, dict)
                else "n/a"
            )
            error = event.message.get("error")
            if error:
                log.debug("%s response (id=%s) ERROR: %s", direction, msg_id, error)
            else:
                log.debug(
                    "%s response (id=%s) stop_reason=%s len=%d",
                    direction, msg_id, stop_reason, msg_len
                )
        elif method == "session/update":
            # For session/update, show update type
            params = event.message.get("params", {})
            update = params.get("update", {})
            update_type = update.get("sessionUpdate", "unknown")
            log.debug(
                "%s %s (id=%s) type=%s len=%d",
                direction, method, msg_id, update_type, msg_len
            )
        else:
            # For other messages, show truncated preview
            preview = msg_str[:200] + "..." if len(msg_str) > 200 else msg_str
            log.debug("%s %s (id=%s) %s", direction, method, msg_id, preview)

    log.info("Setting up stdio connection...")
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

    log.info("Ready to accept ACP requests")

    try:
        await conn.listen()
    except (BrokenPipeError, ConnectionResetError):
        log.info("Pipe closed, shutting down...")
    finally:
        # Ensure connection is properly closed with timeout
        log.info("Connection closed, cleaning up...")
        try:
            await asyncio.wait_for(conn.close(), timeout=2.0)
        except asyncio.TimeoutError:
            log.warning("Connection close timed out, forcing exit")
        except Exception as e:
            log.warning("Error during cleanup: %s", e)


def main() -> None:
    """Run the ActiveContext ACP agent."""
    import asyncio

    from activecontext.config import load_config

    # Expand environment variable references like ${VAR_NAME}
    # This allows acp.json to reference system env vars:
    #   "env": { "OPENAI_API_KEY": "${OPENAI_API_KEY}" }
    _expand_env_vars()

    # Load config before logging so we can use config.logging settings
    config = load_config()

    # Initialize logging with config
    setup_logging(config.logging)

    from activecontext.core.llm.discovery import get_default_model

    log.info("Starting ActiveContext ACP agent...")
    model = get_default_model()
    log.info("Configuration loaded (role=%s, provider=%s, model=%s, budget=%s)",
              config.llm.role or "balanced",
              config.llm.provider or "auto",
              model or "none",
              config.projection.total_budget or "default")

    # Ensure we exit when parent dies
    _setup_parent_death_monitor()

    try:
        asyncio.run(_main())
    finally:
        # Ensure process exits even if there are lingering resources
        log.info("Exiting...")
        os._exit(0)


if __name__ == "__main__":
    main()
