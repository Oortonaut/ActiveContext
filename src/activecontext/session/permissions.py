"""File permission management for the Timeline sandbox.

Provides a PermissionManager that controls file access via configurable
glob patterns. Permissions are defined in config files and hot-reload
automatically when the config changes.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    from activecontext.config.schema import ImportConfig, SandboxConfig

_log = logging.getLogger("activecontext.session.permissions")


@dataclass
class PermissionDenied(Exception):
    """Raised when file access is denied, carrying info for permission request.

    This exception is caught by Timeline to trigger ACP permission requests.
    Unlike PermissionError, it carries metadata needed to ask the user.
    """

    path: str  # Resolved absolute path
    mode: Literal["read", "write"]
    original_path: str  # Path as provided by code

    def __str__(self) -> str:
        return f"Access denied: {self.mode} access to '{self.path}'"


@dataclass
class PermissionRule:
    """A resolved permission rule with absolute pattern."""

    pattern: str  # Resolved glob pattern
    mode: Literal["read", "write", "all"]
    source: str  # "config" or "auto"


@dataclass
class PermissionManager:
    """Manages file permissions for the Timeline sandbox.

    Permissions come from config files only - there's no programmatic
    grant/revoke API. When config files change, call reload() to update.

    Temporary grants can be added via grant_temporary() for session-scoped
    "allow once" permissions from interactive prompts.

    Attributes:
        cwd: Working directory (resolved to absolute path)
        rules: List of permission rules
    """

    cwd: Path
    rules: list[PermissionRule] = field(default_factory=list)
    deny_by_default: bool = True
    allow_absolute: bool = False
    _temporary_grants: set[tuple[str, str]] = field(default_factory=set)  # (path, mode)

    @classmethod
    def from_config(cls, cwd: str, config: SandboxConfig | None) -> PermissionManager:
        """Create a PermissionManager from a SandboxConfig.

        Args:
            cwd: Working directory for the session.
            config: Sandbox configuration (or None for defaults).

        Returns:
            Configured PermissionManager.
        """
        cwd_path = Path(cwd).resolve()
        rules: list[PermissionRule] = []

        if config is None:
            # Default: read-only access to cwd
            rules.append(
                PermissionRule(
                    pattern=str(cwd_path / "**"),
                    mode="read",
                    source="auto",
                )
            )
            return cls(
                cwd=cwd_path,
                rules=rules,
                deny_by_default=True,
                allow_absolute=False,
            )

        # Auto-grant cwd access if configured
        if config.allow_cwd:
            mode: Literal["read", "write", "all"] = (
                "all" if config.allow_cwd_write else "read"
            )
            rules.append(
                PermissionRule(
                    pattern=str(cwd_path / "**"),
                    mode=mode,
                    source="auto",
                )
            )

        # Add explicit file permissions from config
        for perm in config.file_permissions:
            # Resolve pattern relative to cwd
            pattern = perm.pattern
            if not Path(pattern).is_absolute():
                # Convert relative pattern to absolute
                # Handle ./ prefix
                if pattern.startswith("./"):
                    pattern = pattern[2:]
                pattern = str(cwd_path / pattern)

            # Validate mode
            mode_str = perm.mode.lower()
            if mode_str not in ("read", "write", "all"):
                _log.warning("Invalid permission mode '%s', defaulting to 'read'", mode_str)
                mode_str = "read"

            rules.append(
                PermissionRule(
                    pattern=pattern,
                    mode=mode_str,  # type: ignore[arg-type]
                    source="config",
                )
            )

        return cls(
            cwd=cwd_path,
            rules=rules,
            deny_by_default=config.deny_by_default,
            allow_absolute=config.allow_absolute,
        )

    def reload(self, config: SandboxConfig | None) -> None:
        """Reload permissions from a new config.

        Args:
            config: New sandbox configuration.
        """
        new_manager = PermissionManager.from_config(str(self.cwd), config)
        self.rules = new_manager.rules
        self.deny_by_default = new_manager.deny_by_default
        self.allow_absolute = new_manager.allow_absolute
        _log.debug("Permissions reloaded: %d rules", len(self.rules))

    def grant_temporary(self, path: str, mode: str) -> None:
        """Grant temporary access for this session (allow_once).

        Temporary grants are not persisted and are cleared when the
        session ends or the PermissionManager is recreated.

        Args:
            path: Absolute path to grant access to.
            mode: "read" or "write".
        """
        resolved = self._resolve_path(path)
        if resolved:
            self._temporary_grants.add((str(resolved), mode))
            _log.debug("Temporary grant added: %s (%s)", resolved, mode)

    def clear_temporary_grants(self) -> None:
        """Clear all temporary grants."""
        self._temporary_grants.clear()
        _log.debug("Temporary grants cleared")

    def _resolve_path(self, path: str) -> Path | None:
        """Resolve a path and validate it's within allowed scope.

        Args:
            path: Path to resolve (may be relative or absolute).

        Returns:
            Resolved absolute Path, or None if path is not allowed.
        """
        try:
            p = Path(path)

            # Handle relative paths
            if not p.is_absolute():
                p = self.cwd / p

            # Resolve symlinks and normalize
            resolved = p.resolve()

            # Check if absolute paths are allowed
            if not self.allow_absolute:
                # Path must be within cwd
                try:
                    resolved.relative_to(self.cwd)
                except ValueError:
                    _log.debug("Path outside cwd denied: %s", resolved)
                    return None

            return resolved

        except (ValueError, OSError) as e:
            _log.debug("Path resolution failed for '%s': %s", path, e)
            return None

    def check_access(self, path: str, mode: Literal["read", "write"]) -> bool:
        """Check if access to a path is permitted.

        Args:
            path: File path to check.
            mode: "read" or "write".

        Returns:
            True if access is permitted.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return False

        resolved_str = str(resolved)

        # Check temporary grants first (from "allow once" prompts)
        if (resolved_str, mode) in self._temporary_grants:
            return True

        # Check against rules (first match wins)
        for rule in self.rules:
            if fnmatch.fnmatch(resolved_str, rule.pattern):
                # Check mode compatibility
                if rule.mode == "all":
                    return True
                if rule.mode == mode:
                    return True
                if mode == "read" and rule.mode == "write":
                    # write permission doesn't grant read
                    continue
                # Mode doesn't match, continue checking other rules

        # No matching rule found
        if self.deny_by_default:
            _log.debug("Access denied (no matching rule): %s (%s)", path, mode)
            return False

        # Allow by default if deny_by_default is False
        return True

    def list_permissions(self) -> list[dict[str, Any]]:
        """List all permission rules for inspection.

        Returns:
            List of rule dictionaries with pattern, mode, and source.
        """
        return [
            {
                "pattern": rule.pattern,
                "mode": rule.mode,
                "source": rule.source,
            }
            for rule in self.rules
        ]


def make_safe_open(permission_manager: PermissionManager) -> Any:
    """Create a permission-checked wrapper around the built-in open().

    Args:
        permission_manager: PermissionManager to use for access checks.

    Returns:
        A wrapped open() function that checks permissions.
    """
    import builtins

    original_open = builtins.open

    def safe_open(
        file: Any,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Permission-checked open() wrapper.

        Raises:
            PermissionDenied: If access to the file is not permitted.
                This exception carries metadata for permission request prompts.
        """
        # Determine access mode
        access_mode: Literal["read", "write"] = "read"
        if any(c in mode for c in "wax+"):
            access_mode = "write"

        # Convert file to string path for checking
        file_path = str(file) if not isinstance(file, (int, bytes)) else None

        if file_path is not None and not permission_manager.check_access(file_path, access_mode):
            # Raise PermissionDenied with metadata for ACP permission request
            resolved = permission_manager._resolve_path(file_path)
            raise PermissionDenied(
                path=str(resolved) if resolved else file_path,
                mode=access_mode,
                original_path=file_path,
            )

        return original_open(file, mode, *args, **kwargs)

    return safe_open


def write_permission_to_config(cwd: Path, pattern: str, mode: str) -> None:
    """Append a permission rule to .ac/config.yaml.

    Creates the config file and directory if they don't exist.
    If the file exists, the new rule is appended to sandbox.file_permissions.

    Args:
        cwd: Working directory (project root).
        pattern: Glob pattern for the permission (e.g., "./data/file.txt").
        mode: "read", "write", or "all".
    """
    config_path = cwd / ".ac" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Ensure sandbox.file_permissions exists
    data.setdefault("sandbox", {})
    data["sandbox"].setdefault("file_permissions", [])

    # Check if rule already exists
    existing_patterns = {
        p.get("pattern")
        for p in data["sandbox"]["file_permissions"]
        if isinstance(p, dict)
    }
    if pattern in existing_patterns:
        _log.debug("Permission rule already exists: %s", pattern)
        return

    # Add new permission
    data["sandbox"]["file_permissions"].append({
        "pattern": pattern,
        "mode": mode,
    })

    # Write back with proper YAML formatting
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    _log.info("Added permission to %s: %s (%s)", config_path, pattern, mode)


# =============================================================================
# Shell Permission System
# =============================================================================


@dataclass
class ShellPermissionDenied(Exception):
    """Raised when shell command is denied, carrying info for permission request.

    This exception is caught by Timeline to trigger ACP permission requests.
    Unlike PermissionError, it carries metadata needed to ask the user.
    """

    command: str  # The command (e.g., "rm", "git")
    full_command: str  # Full command string for display
    command_args: list[str] | None = None  # Command arguments (renamed to avoid conflict with Exception.args)

    def __str__(self) -> str:
        return f"Shell command denied: {self.full_command}"


@dataclass
class ShellPermissionRule:
    """A resolved shell permission rule."""

    pattern: str  # Glob pattern to match against full command
    allow: bool  # True to allow, False to deny


@dataclass
class ShellPermissionManager:
    """Manages shell command permissions for the Timeline sandbox.

    Permissions come from config files. Rules are matched against the full
    command string (command + args). First matching rule wins.

    Temporary grants can be added via grant_temporary() for session-scoped
    "allow once" permissions from interactive prompts.
    """

    rules: list[ShellPermissionRule] = field(default_factory=list)
    deny_by_default: bool = True
    _temporary_grants: set[str] = field(default_factory=set)  # full command strings

    @classmethod
    def from_config(cls, config: SandboxConfig | None) -> ShellPermissionManager:
        """Create a ShellPermissionManager from a SandboxConfig.

        Args:
            config: Sandbox configuration (or None for defaults).

        Returns:
            Configured ShellPermissionManager.
        """
        if config is None:
            # Default: deny all shell commands
            return cls(rules=[], deny_by_default=True)

        rules = [
            ShellPermissionRule(pattern=p.pattern, allow=p.allow)
            for p in config.shell_permissions
        ]

        return cls(
            rules=rules,
            deny_by_default=config.shell_deny_by_default,
        )

    def reload(self, config: SandboxConfig | None) -> None:
        """Reload permissions from a new config.

        Args:
            config: New sandbox configuration.
        """
        new_manager = ShellPermissionManager.from_config(config)
        self.rules = new_manager.rules
        self.deny_by_default = new_manager.deny_by_default
        _log.debug("Shell permissions reloaded: %d rules", len(self.rules))

    def check_access(self, command: str, args: list[str] | None = None) -> bool:
        """Check if a shell command is permitted.

        Args:
            command: The command to execute (e.g., "git", "npm").
            args: Optional command arguments.

        Returns:
            True if the command is permitted.
        """
        full_command = self._build_command_string(command, args)

        # Check temporary grants first (from "allow once" prompts)
        if full_command in self._temporary_grants:
            return True

        # Check against rules (first match wins)
        for rule in self.rules:
            if fnmatch.fnmatch(full_command, rule.pattern):
                return rule.allow

        # No matching rule found
        if self.deny_by_default:
            _log.debug("Shell command denied (no matching rule): %s", full_command)
            return False

        # Allow by default if deny_by_default is False
        return True

    def grant_temporary(self, command: str, args: list[str] | None = None) -> None:
        """Grant temporary access for this session (allow_once).

        Temporary grants are not persisted and are cleared when the
        session ends or the ShellPermissionManager is recreated.

        Args:
            command: The command to allow.
            args: Command arguments.
        """
        full_command = self._build_command_string(command, args)
        self._temporary_grants.add(full_command)
        _log.debug("Temporary shell grant added: %s", full_command)

    def clear_temporary_grants(self) -> None:
        """Clear all temporary grants."""
        self._temporary_grants.clear()
        _log.debug("Temporary shell grants cleared")

    def _build_command_string(self, command: str, args: list[str] | None) -> str:
        """Build full command string for matching.

        Args:
            command: The command name.
            args: Optional command arguments.

        Returns:
            Full command string (e.g., "git status", "npm run build").
        """
        if args:
            return f"{command} {' '.join(args)}"
        return command

    def list_permissions(self) -> list[dict[str, Any]]:
        """List all shell permission rules for inspection.

        Returns:
            List of rule dictionaries with pattern and allow flag.
        """
        return [
            {
                "pattern": rule.pattern,
                "allow": rule.allow,
            }
            for rule in self.rules
        ]


def write_shell_permission_to_config(
    cwd: Path, command: str, args: list[str] | None = None
) -> None:
    """Append a shell permission rule to .ac/config.yaml.

    Creates the config file and directory if they don't exist.
    If the file exists, the new rule is appended to sandbox.shell_permissions.

    Args:
        cwd: Working directory (project root).
        command: The command to allow.
        args: Command arguments (used to build the pattern).
    """
    config_path = cwd / ".ac" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Ensure sandbox.shell_permissions exists
    data.setdefault("sandbox", {})
    data["sandbox"].setdefault("shell_permissions", [])

    # Build pattern - use the full command string
    if args:
        pattern = f"{command} {' '.join(args)}"
    else:
        pattern = command

    # Check if rule already exists
    existing_patterns = {
        p.get("pattern")
        for p in data["sandbox"]["shell_permissions"]
        if isinstance(p, dict)
    }
    if pattern in existing_patterns:
        _log.debug("Shell permission rule already exists: %s", pattern)
        return

    # Add new permission
    data["sandbox"]["shell_permissions"].append({
        "pattern": pattern,
        "allow": True,
    })

    # Write back with proper YAML formatting
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    _log.info("Added shell permission to %s: %s", config_path, pattern)


@dataclass
class ImportDenied(Exception):
    """Raised when module import is denied.

    This exception indicates the module is not in the whitelist.
    """

    module: str  # Full module name that was requested
    top_level: str  # Top-level module name

    def __str__(self) -> str:
        return f"Import denied: '{self.module}' is not in the allowed modules whitelist"


@dataclass
class ImportGuard:
    """Manages import whitelist for the Timeline sandbox.

    Controls which modules can be imported during code execution.
    Supports whitelisting top-level modules and optionally their submodules.
    """

    allowed_modules: set[str] = field(default_factory=set)
    allow_submodules: bool = True
    allow_all: bool = False

    @classmethod
    def from_config(cls, config: ImportConfig | None) -> ImportGuard:
        """Create an ImportGuard from an ImportConfig.

        Args:
            config: Import configuration (or None for deny-all).

        Returns:
            Configured ImportGuard.
        """
        from activecontext.config.schema import ImportConfig

        if config is None:
            config = ImportConfig()

        return cls(
            allowed_modules=set(config.allowed_modules),
            allow_submodules=config.allow_submodules,
            allow_all=config.allow_all,
        )

    def is_allowed(self, module_name: str) -> bool:
        """Check if a module import is allowed.

        Args:
            module_name: Full module name (e.g., "os.path", "json").

        Returns:
            True if the import is allowed.
        """
        if self.allow_all:
            return True

        if not self.allowed_modules:
            # Empty whitelist means nothing is allowed
            return False

        # Get the top-level module
        top_level = module_name.split(".")[0]

        # Check exact match first
        if module_name in self.allowed_modules:
            return True

        # Check if top-level module is allowed
        if top_level in self.allowed_modules:
            # If submodules are allowed, permit any submodule
            if self.allow_submodules:
                return True
            # Otherwise only exact match counts (already checked above)
            return module_name == top_level

        return False

    def add_module(self, module: str) -> None:
        """Add a module to the whitelist.

        Args:
            module: Module name to allow.
        """
        self.allowed_modules.add(module)
        _log.debug("Added module to import whitelist: %s", module)

    def remove_module(self, module: str) -> None:
        """Remove a module from the whitelist.

        Args:
            module: Module name to disallow.
        """
        self.allowed_modules.discard(module)
        _log.debug("Removed module from import whitelist: %s", module)

    def list_allowed(self) -> list[str]:
        """List all allowed modules.

        Returns:
            Sorted list of allowed module names.
        """
        return sorted(self.allowed_modules)


def make_safe_import(import_guard: ImportGuard) -> Any:
    """Create a whitelist-checked wrapper around __import__.

    Args:
        import_guard: ImportGuard to use for whitelist checks.

    Returns:
        A wrapped __import__ function that checks the whitelist.
    """
    import builtins

    original_import = builtins.__import__

    def safe_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        """Whitelist-checked __import__ wrapper.

        Raises:
            ImportDenied: If the module is not in the whitelist.
        """
        # For relative imports (level > 0), we need the package context
        # These are typically internal to an already-imported module
        if level > 0:
            # Relative imports within allowed modules should work
            # The calling module was already validated when first imported
            return original_import(name, globals, locals, fromlist, level)

        # Check the main module being imported
        if not import_guard.is_allowed(name):
            top_level = name.split(".")[0]
            raise ImportDenied(module=name, top_level=top_level)

        # Also check fromlist items if present (e.g., "from os import path")
        if fromlist:
            for item in fromlist:
                full_name = f"{name}.{item}"
                # Only check if it looks like a submodule, not an attribute
                # We can't easily distinguish, so we check if it's allowed
                # If not explicitly blocked, the import proceeds and fails
                # naturally if the attribute doesn't exist
                if not import_guard.is_allowed(full_name) and not import_guard.is_allowed(name):
                    raise ImportDenied(module=full_name, top_level=name.split(".")[0])

        return original_import(name, globals, locals, fromlist, level)

    return safe_import
