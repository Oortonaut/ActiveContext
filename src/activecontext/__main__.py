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
    """Async entry point."""
    import acp

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

    log.info("Ready to accept ACP requests")
    await acp.run_agent(agent, use_unstable_protocol=True)


def main() -> None:
    """Run the ActiveContext ACP agent."""
    import asyncio

    from activecontext.config import load_config

    # When stdin is piped (IDE or echo), silence stderr and root logger
    # to prevent tracebacks from interfering with ACP protocol
    if not sys.stdin.isatty():
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115 - intentionally kept open for process lifetime
        import logging
        logging.getLogger().addHandler(logging.NullHandler())
        logging.getLogger().setLevel(logging.CRITICAL + 1)  # Silence everything

    _expand_env_vars()
    config = load_config()
    setup_logging(config.logging)

    log.info("Starting ActiveContext ACP agent...")
    asyncio.run(_main())


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

    main()
