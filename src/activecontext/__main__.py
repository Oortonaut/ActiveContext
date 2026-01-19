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

    # Track last message time for idle timeout
    import time
    last_message_time = time.monotonic()
    IDLE_TIMEOUT = 30.0  # Exit after 30 seconds of no messages

    def log_message(event: StreamEvent) -> None:
        """Log all ACP messages for debugging."""
        nonlocal last_message_time
        last_message_time = time.monotonic()
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

    # Run listen() with an idle timeout watchdog
    async def watchdog() -> None:
        """Exit if no messages received for IDLE_TIMEOUT seconds."""
        while True:
            await asyncio.sleep(5.0)  # Check every 5 seconds
            idle_time = time.monotonic() - last_message_time
            if idle_time > IDLE_TIMEOUT:
                log.info("Watchdog: idle for %.1fs, exiting", idle_time)
                os._exit(0)

    # Start watchdog and listen concurrently
    listen_task = asyncio.create_task(conn.listen())
    watchdog_task = asyncio.create_task(watchdog())

    try:
        # Wait for either to complete
        done, pending = await asyncio.wait(
            [listen_task, watchdog_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        # Check results
        for task in done:
            try:
                task.result()
            except Exception as e:
                log.info("Task ended with: %s", e)

    except Exception as e:
        log.info("Wait ended with: %s, exiting", e)

    log.info("Exiting immediately")
    os._exit(0)


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

    try:
        asyncio.run(_main())
    finally:
        # Ensure process exits even if there are lingering resources
        log.info("Exiting...")
        os._exit(0)


if __name__ == "__main__":
    main()
