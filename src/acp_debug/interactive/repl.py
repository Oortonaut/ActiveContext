"""Interactive REPL for acp-debug."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.console import Console

from acp_debug.interactive.commands import CommandHandler

if TYPE_CHECKING:
    from pathlib import Path

    from acp_debug.config import Config
    from acp_debug.extension.chain import ExtensionChain

console = Console()


class InteractiveRepl:
    """Interactive REPL with slash commands."""

    def __init__(
        self,
        config: Config,
        chain: ExtensionChain,
        history_file: Path | None = None,
    ) -> None:
        self.config = config
        self.chain = chain
        self.commands = CommandHandler(config, chain)
        self._running = False

        # Setup prompt session
        history = FileHistory(str(history_file)) if history_file else None
        self.session: PromptSession[str] = PromptSession(
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
        )

    async def run(self) -> None:
        """Run the interactive REPL."""
        self._running = True

        console.print("[bold]ACP Debug[/bold] v0.1.0 - Interactive Mode")
        console.print("Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.\n")

        while self._running:
            try:
                # Get input
                line = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.prompt("acp> "),
                )

                line = line.strip()
                if not line:
                    continue

                # Handle commands
                if line.startswith("/"):
                    await self.commands.handle(line)
                    if line == "/quit":
                        break
                else:
                    console.print("[dim]Unknown input. Type /help for commands.[/dim]")

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

        self._running = False

    def stop(self) -> None:
        """Stop the REPL."""
        self._running = False
