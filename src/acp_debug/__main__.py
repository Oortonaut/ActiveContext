"""CLI entry point for acp-debug."""

import sys


def main() -> int:
    """Main entry point for acp-debug CLI."""
    from acp_debug.cli import run_cli

    return run_cli(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
