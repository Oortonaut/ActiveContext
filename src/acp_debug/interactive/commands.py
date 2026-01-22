"""Slash command handlers for interactive mode."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from acp_debug.config import Config
    from acp_debug.extension.chain import ExtensionChain

console = Console()


class CommandHandler:
    """Handles slash commands in interactive mode."""

    def __init__(self, config: Config, chain: ExtensionChain) -> None:
        self.config = config
        self.chain = chain
        self._recording = False
        self._recording_file: Path | None = None
        self._paused = False

    async def handle(self, line: str) -> None:
        """Handle a slash command."""
        parts = shlex.split(line)
        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:]

        handlers = {
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/sessions": self._cmd_sessions,
            "/extensions": self._cmd_extensions,
            "/reload": self._cmd_reload,
            "/inject": self._cmd_inject,
            "/record": self._cmd_record,
            "/replay": self._cmd_replay,
            "/pause": self._cmd_pause,
            "/resume": self._cmd_resume,
            "/drop": self._cmd_drop,
            "/log": self._cmd_log,
            "/quit": self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler:
            await handler(args)
        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
            console.print("Type [bold]/help[/bold] for available commands.")

    async def _cmd_help(self, args: list[str]) -> None:
        """Show available commands."""
        table = Table(title="Available Commands")
        table.add_column("Command", style="bold")
        table.add_column("Description")

        commands = [
            ("/help", "Show this help message"),
            ("/status", "Show current status"),
            ("/sessions", "List active sessions"),
            ("/extensions", "List loaded extensions"),
            ("/reload", "Hot-reload extensions"),
            ("/inject <json>", "Send raw JSON-RPC message"),
            ("/record <file>", "Start recording to file"),
            ("/replay <file> [hooks|no-hooks]", "Replay recorded session"),
            ("/pause", "Pause message forwarding"),
            ("/resume", "Resume message forwarding"),
            ("/drop <pattern>", "Drop messages matching pattern"),
            ("/log <method> [dest] || [off]", "Configure per-method logging"),
            ("/quit", "Exit debugger"),
        ]

        for cmd, desc in commands:
            table.add_row(cmd, desc)

        console.print(table)

    async def _cmd_status(self, args: list[str]) -> None:
        """Show current status."""
        console.print("[bold]Status:[/bold]")
        console.print(f"  Extensions: {len(self.chain.extensions)} loaded")
        console.print(f"  Recording: {'yes' if self._recording else 'no'}")
        console.print(f"  Paused: {'yes' if self._paused else 'no'}")

        # Show log config
        if self.config.log_config:
            console.print("  Logging:")
            for method, dest in self.config.log_config.items():
                console.print(f"    {method} → {dest}")

    async def _cmd_sessions(self, args: list[str]) -> None:
        """List active sessions."""
        sessions_found = False
        for ext in self.chain.extensions:
            if ext.sessions:
                sessions_found = True
                table = Table(title="Active Sessions")
                table.add_column("Session ID")
                table.add_column("Mode")
                table.add_column("Model")
                table.add_column("CWD")

                for sid, state in ext.sessions.items():
                    table.add_row(
                        sid,
                        state.mode or "-",
                        state.model or "-",
                        state.cwd or "-",
                    )

                console.print(table)
                break

        if not sessions_found:
            console.print("[dim]No active sessions[/dim]")

    async def _cmd_extensions(self, args: list[str]) -> None:
        """List loaded extensions."""
        table = Table(title="Loaded Extensions")
        table.add_column("Class")
        table.add_column("Overridden Methods")

        from acp_debug.extension.base import ACPExtension

        base_methods = set(dir(ACPExtension))

        for ext in self.chain.extensions:
            cls_name = ext.__class__.__name__

            # Find overridden methods
            overridden = []
            for name in dir(ext):
                if name.startswith("_"):
                    continue
                if name in base_methods:
                    # Check if actually overridden
                    ext_method = getattr(ext.__class__, name, None)
                    base_method = getattr(ACPExtension, name, None)
                    if ext_method is not base_method:
                        overridden.append(name)

            table.add_row(cls_name, ", ".join(overridden) or "-")

        console.print(table)

    async def _cmd_reload(self, args: list[str]) -> None:
        """Reload extensions."""
        from acp_debug.extension.loader import reload_extensions

        new_extensions = reload_extensions(self.config, self.chain.extensions)
        self.chain.extensions = new_extensions
        self.chain.initialize_all()

        console.print(f"[green]Reloaded {len(new_extensions)} extension(s)[/green]")

    async def _cmd_inject(self, args: list[str]) -> None:
        """Inject a raw JSON-RPC message."""
        if not args:
            console.print("[red]Usage: /inject <json>[/red]")
            return

        try:
            json_str = " ".join(args)
            data = json.loads(json_str)
            console.print(f"[yellow]Inject not yet implemented. Parsed: {data}[/yellow]")
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON: {e}[/red]")

    async def _cmd_record(self, args: list[str]) -> None:
        """Start or stop recording."""
        if not args:
            if self._recording:
                console.print(f"[green]Recording to: {self._recording_file}[/green]")
            else:
                console.print("[dim]Not recording. Usage: /record <file>[/dim]")
            return

        file_path = Path(args[0])
        self._recording = True
        self._recording_file = file_path
        console.print(f"[green]Recording to: {file_path}[/green]")

    async def _cmd_replay(self, args: list[str]) -> None:
        """Replay a recorded session."""
        if not args:
            console.print("[red]Usage: /replay <file> [hooks|no-hooks][/red]")
            return

        file_path = Path(args[0])
        use_hooks = True

        if len(args) > 1:
            if args[1] == "no-hooks":
                use_hooks = False
            elif args[1] != "hooks":
                console.print(f"[red]Unknown option: {args[1]}[/red]")
                return

        console.print(
            f"[yellow]Replay not yet implemented: {file_path} (hooks={use_hooks})[/yellow]"
        )

    async def _cmd_pause(self, args: list[str]) -> None:
        """Pause message forwarding."""
        self._paused = True
        console.print("[yellow]Message forwarding paused[/yellow]")

    async def _cmd_resume(self, args: list[str]) -> None:
        """Resume message forwarding."""
        self._paused = False
        console.print("[green]Message forwarding resumed[/green]")

    async def _cmd_drop(self, args: list[str]) -> None:
        """Drop messages matching pattern."""
        if not args:
            console.print("[red]Usage: /drop <pattern>[/red]")
            return

        pattern = args[0]
        console.print(f"[yellow]Drop not yet implemented: {pattern}[/yellow]")

    async def _cmd_log(self, args: list[str]) -> None:
        """Configure per-method logging."""
        if not args:
            # Show current config
            console.print("[bold]Logging configuration:[/bold]")
            if self.config.log_config:
                for method, dest in self.config.log_config.items():
                    console.print(f"  {method} → {dest}")
            else:
                console.print("  [dim](no logging configured)[/dim]")
            return

        method = args[0]

        if len(args) == 1:
            # Show config for specific method
            dest = self.config.log_config.get(method, "(off)")
            console.print(f"  {method} → {dest}")
            return

        # Set logging destination
        dest = args[1]

        if dest == "off":
            self.config.log_config.pop(method, None)
            console.print(f"[dim]Logging disabled for {method}[/dim]")
        elif dest in ("stdout", "stderr"):
            self.config.log_config[method] = dest
            console.print(f"[green]{method} → {dest}[/green]")
        elif dest == "file":
            if len(args) < 3:
                console.print("[red]Usage: /log <method> file <path>[/red]")
                return
            file_path = args[2]
            self.config.log_config[method] = f"file:{file_path}"
            console.print(f"[green]{method} → file:{file_path}[/green]")
        else:
            console.print(f"[red]Unknown destination: {dest}[/red]")
            console.print("Options: stdout, stderr, file <path>, off")

    async def _cmd_quit(self, args: list[str]) -> None:
        """Exit the debugger."""
        console.print("[dim]Exiting...[/dim]")
