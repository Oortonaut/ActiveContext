"""Entry point for running ActiveContext as an ACP agent.

Usage:
    python -m activecontext

    # For testing with a file:
    cat tests/fixtures/init_and_session.jsonl | python -m activecontext

This starts the ACP agent listening on stdin/stdout for JSON-RPC
messages from an ACP client (Rider, Zed, etc.).
"""

import sys


def _log(msg: str) -> None:
    """Log to stderr for debugging."""
    print(f"[activecontext] {msg}", file=sys.stderr, flush=True)


def main() -> None:
    """Run the ActiveContext ACP agent."""
    import asyncio

    _log("Starting...")

    import acp

    _log("Importing agent...")
    from activecontext.transport.acp.agent import create_agent

    _log("Creating agent...")
    agent = create_agent()

    _log(f"Agent ready, model={agent._current_model_id}")
    asyncio.run(acp.run_agent(agent, use_unstable_protocol=True))


if __name__ == "__main__":
    main()
