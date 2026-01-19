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
    """Async entry point - use simple acp.run_agent() like the original."""
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

    # Use simple acp.run_agent() - this is what worked in the original
    await acp.run_agent(agent, use_unstable_protocol=True)


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
