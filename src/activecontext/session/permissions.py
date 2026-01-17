"""File permission management for the Timeline sandbox.

Provides a PermissionManager that controls file access via configurable
glob patterns. Permissions are defined in config files and hot-reload
automatically when the config changes.
"""

from __future__ import annotations

import fnmatch
import logging
import re
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


# =============================================================================
# Typed Placeholder Pattern Matching
# =============================================================================


@dataclass
class PatternSegment:
    """Base class for pattern segments."""

    pass


@dataclass
class LiteralSegment(PatternSegment):
    """Exact string match."""

    value: str


@dataclass
class PlaceholderSegment(PatternSegment):
    """Typed placeholder: {name:type} or {:type}."""

    name: str | None  # None for anonymous {:type}
    type: str  # "dir", "r", "w", "args", "str", etc.


@dataclass
class GlobSegment(PatternSegment):
    """Legacy glob wildcard: *"""

    pass


# Regex to match placeholders: {name:type} or {:type}
_PLACEHOLDER_PATTERN = re.compile(r"\{(?:([a-zA-Z_][a-zA-Z0-9_]*)?):([a-zA-Z_][a-zA-Z0-9_]*)\}")


def parse_pattern(pattern: str) -> list[PatternSegment]:
    """Parse pattern into segments.

    Examples:
        "rm -rf {target:dir}" -> [Literal("rm"), Literal("-rf"), Placeholder("target", "dir")]
        "git add {:args} {file:r}" -> [Literal("git"), Literal("add"), Placeholder(None, "args"), Placeholder("file", "r")]
        "npm run *" -> [Literal("npm"), Literal("run"), Glob()]

    Args:
        pattern: The pattern string to parse.

    Returns:
        List of PatternSegment instances.
    """
    segments: list[PatternSegment] = []

    # Split pattern on whitespace, preserving structure
    tokens = pattern.split()

    for token in tokens:
        # Check if token is a placeholder
        match = _PLACEHOLDER_PATTERN.fullmatch(token)
        if match:
            name = match.group(1)  # May be None for {:type}
            type_name = match.group(2)
            segments.append(PlaceholderSegment(name=name, type=type_name))
        elif token == "*":
            # Legacy glob wildcard
            segments.append(GlobSegment())
        else:
            # Literal string
            segments.append(LiteralSegment(value=token))

    return segments


def is_typed_pattern(pattern: str) -> bool:
    """Check if a pattern uses typed placeholders.

    Args:
        pattern: The pattern string to check.

    Returns:
        True if the pattern contains typed placeholders.
    """
    return bool(_PLACEHOLDER_PATTERN.search(pattern))


@dataclass
class TypeValidator:
    """Validates captured values against their declared types."""

    cwd: Path
    permission_manager: PermissionManager | None = None

    def validate(self, value: str, type_name: str) -> bool:
        """Validate a captured value against its type.

        Args:
            value: The value to validate.
            type_name: The type name (e.g., "dir", "r", "path").

        Returns:
            True if the value is valid for the type.
        """
        match type_name:
            case "dir":
                path = self._resolve_path(value)
                return path.is_dir() if path else False

            case "mdir":
                path = self._resolve_path(value)
                if not path or not path.is_dir():
                    return False
                # Check write permission
                if self.permission_manager:
                    return self.permission_manager.check_access(str(path), "write")
                return True

            case "r" | "read":
                path = self._resolve_path(value)
                if not path or not path.exists():
                    return False
                if self.permission_manager:
                    return self.permission_manager.check_access(str(path), "read")
                return True

            case "w" | "write":
                if self.permission_manager:
                    path = self._resolve_path(value)
                    return (
                        self.permission_manager.check_access(str(path), "write")
                        if path
                        else False
                    )
                return True

            case "path":
                path = self._resolve_path(value)
                return path.exists() if path else False

            case "str":
                return True  # Any string is valid

            case "int":
                return value.lstrip("-").isdigit()

            case "args":
                return True  # Args captures multiple tokens, always valid

            case _:
                _log.warning("Unknown placeholder type: %s", type_name)
                return False  # Unknown type

    def _resolve_path(self, value: str) -> Path | None:
        """Resolve a path relative to cwd.

        Args:
            value: The path string to resolve.

        Returns:
            Resolved absolute Path, or None if resolution fails.
        """
        try:
            p = Path(value)
            if not p.is_absolute():
                p = self.cwd / p
            return p.resolve()
        except (ValueError, OSError) as e:
            _log.debug("Path resolution failed for '%s': %s", value, e)
            return None


@dataclass
class MatchResult:
    """Result of pattern matching."""

    matched: bool
    captures: dict[str, str] = field(default_factory=dict)  # name -> value


class PatternMatcher:
    """Matches command strings against typed patterns."""

    def __init__(
        self, cwd: Path, permission_manager: PermissionManager | None = None
    ) -> None:
        """Initialize the pattern matcher.

        Args:
            cwd: Working directory for path resolution.
            permission_manager: Optional PermissionManager for file permission checks.
        """
        self.cwd = cwd
        self.validator = TypeValidator(cwd, permission_manager)

    def match(
        self, pattern: str, command: str, args: list[str] | None
    ) -> MatchResult:
        """Match a command against a pattern.

        Args:
            pattern: The pattern to match against.
            command: The command name (e.g., "git", "rm").
            args: Optional command arguments.

        Returns:
            MatchResult with matched=True if the pattern matches.
        """
        # Check if pattern uses typed placeholders
        if is_typed_pattern(pattern):
            return self._match_typed(pattern, command, args)
        else:
            # Fall back to legacy fnmatch glob
            full_cmd = f"{command} {' '.join(args or [])}" if args else command
            if fnmatch.fnmatch(full_cmd, pattern):
                return MatchResult(matched=True)
            return MatchResult(matched=False)

    def _match_typed(
        self, pattern: str, command: str, args: list[str] | None
    ) -> MatchResult:
        """Match using typed placeholder system.

        Args:
            pattern: The pattern with typed placeholders.
            command: The command name.
            args: Command arguments.

        Returns:
            MatchResult with matched status and captured values.
        """
        segments = parse_pattern(pattern)
        tokens = [command] + (args or [])
        captures: dict[str, str] = {}

        seg_idx = 0
        tok_idx = 0

        while seg_idx < len(segments):
            segment = segments[seg_idx]

            if isinstance(segment, LiteralSegment):
                # Must have a token to match
                if tok_idx >= len(tokens):
                    return MatchResult(matched=False)
                if tokens[tok_idx] != segment.value:
                    return MatchResult(matched=False)
                tok_idx += 1
                seg_idx += 1

            elif isinstance(segment, PlaceholderSegment):
                if segment.type == "args":
                    # Greedy: capture all remaining tokens until next literal segment
                    # or end of segments
                    next_literal = None
                    for i in range(seg_idx + 1, len(segments)):
                        next_seg = segments[i]
                        if isinstance(next_seg, LiteralSegment):
                            next_literal = next_seg.value
                            break

                    if next_literal:
                        # Find where the next literal appears in tokens
                        captured_tokens = []
                        while tok_idx < len(tokens) and tokens[tok_idx] != next_literal:
                            captured_tokens.append(tokens[tok_idx])
                            tok_idx += 1
                        if segment.name:
                            captures[segment.name] = " ".join(captured_tokens)
                    else:
                        # Capture all remaining tokens
                        captured_tokens = tokens[tok_idx:]
                        if segment.name:
                            captures[segment.name] = " ".join(captured_tokens)
                        tok_idx = len(tokens)
                    seg_idx += 1
                else:
                    # Must have a token to match
                    if tok_idx >= len(tokens):
                        return MatchResult(matched=False)
                    value = tokens[tok_idx]
                    if not self.validator.validate(value, segment.type):
                        return MatchResult(matched=False)
                    if segment.name:
                        captures[segment.name] = value
                    tok_idx += 1
                    seg_idx += 1

            elif isinstance(segment, GlobSegment):
                # Legacy glob: consume rest and match
                return MatchResult(matched=True, captures=captures)

            else:
                # Unknown segment type
                return MatchResult(matched=False)

        # Check we consumed all tokens
        matched = tok_idx == len(tokens)
        return MatchResult(matched=matched, captures=captures)


@dataclass
class ShellPermissionManager:
    """Manages shell command permissions for the Timeline sandbox.

    Permissions come from config files. Rules are matched against the full
    command string (command + args). First matching rule wins.

    Supports both legacy glob patterns and typed placeholders:
    - Legacy: "git *" matches any git command
    - Typed: "rm -rf {target:dir}" only matches if target is a directory

    Temporary grants can be added via grant_temporary() for session-scoped
    "allow once" permissions from interactive prompts.
    """

    rules: list[ShellPermissionRule] = field(default_factory=list)
    deny_by_default: bool = True
    cwd: Path | None = None  # For typed pattern path resolution
    permission_manager: PermissionManager | None = None  # For file permission delegation
    _temporary_grants: set[str] = field(default_factory=set)  # full command strings
    _matcher: PatternMatcher | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize the pattern matcher if cwd is provided."""
        if self.cwd:
            self._matcher = PatternMatcher(self.cwd, self.permission_manager)

    @classmethod
    def from_config(
        cls,
        config: SandboxConfig | None,
        cwd: Path | None = None,
        permission_manager: PermissionManager | None = None,
    ) -> ShellPermissionManager:
        """Create a ShellPermissionManager from a SandboxConfig.

        Args:
            config: Sandbox configuration (or None for defaults).
            cwd: Working directory for typed pattern path resolution.
            permission_manager: Optional PermissionManager for file permission delegation.

        Returns:
            Configured ShellPermissionManager.
        """
        if config is None:
            # Default: deny all shell commands
            return cls(
                rules=[],
                deny_by_default=True,
                cwd=cwd,
                permission_manager=permission_manager,
            )

        rules = [
            ShellPermissionRule(pattern=p.pattern, allow=p.allow)
            for p in config.shell_permissions
        ]

        return cls(
            rules=rules,
            deny_by_default=config.shell_deny_by_default,
            cwd=cwd,
            permission_manager=permission_manager,
        )

    def reload(self, config: SandboxConfig | None) -> None:
        """Reload permissions from a new config.

        Args:
            config: New sandbox configuration.
        """
        new_manager = ShellPermissionManager.from_config(
            config, self.cwd, self.permission_manager
        )
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
        result = self.check_access_with_captures(command, args)
        return result[0]

    def check_access_with_captures(
        self, command: str, args: list[str] | None = None
    ) -> tuple[bool, dict[str, str]]:
        """Check if a shell command is permitted and return captures.

        This method is useful for getting named captures from typed placeholders
        to display in permission prompts.

        Args:
            command: The command to execute (e.g., "git", "npm").
            args: Optional command arguments.

        Returns:
            Tuple of (allowed, captures) where captures is a dict of named values.
        """
        full_command = self._build_command_string(command, args)

        # Check temporary grants first (from "allow once" prompts)
        if full_command in self._temporary_grants:
            return (True, {})

        # Check against rules (first match wins)
        for rule in self.rules:
            if self._matcher:
                # Use typed pattern matching
                result = self._matcher.match(rule.pattern, command, args)
                if result.matched:
                    return (rule.allow, result.captures)
            else:
                # Fallback to simple fnmatch if no matcher
                if fnmatch.fnmatch(full_command, rule.pattern):
                    return (rule.allow, {})

        # No matching rule found
        if self.deny_by_default:
            _log.debug("Shell command denied (no matching rule): %s", full_command)
            return (False, {})

        # Allow by default if deny_by_default is False
        return (True, {})

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

    Temporary grants can be added via grant_temporary() for session-scoped
    "allow once" permissions from interactive prompts.
    """

    allowed_modules: set[str] = field(default_factory=set)
    allow_submodules: bool = True
    allow_all: bool = False
    _temporary_grants: set[str] = field(default_factory=set)  # module names
    _temporary_submodule_grants: set[str] = field(default_factory=set)  # modules with submodule access

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

        # Get the top-level module
        top_level = module_name.split(".")[0]

        # Check temporary grants first (from "allow once" prompts)
        if module_name in self._temporary_grants:
            return True
        if top_level in self._temporary_grants:
            # Check if this temporary grant includes submodules
            if top_level in self._temporary_submodule_grants:
                return True
            # Otherwise only exact match counts
            if module_name == top_level:
                return True

        if not self.allowed_modules:
            # Empty whitelist means nothing is allowed (unless temp granted)
            return False

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

    def grant_temporary(self, module: str, include_submodules: bool = False) -> None:
        """Grant temporary access to a module (allow_once).

        Temporary grants are not persisted and are cleared when the
        session ends or the ImportGuard is recreated.

        Args:
            module: Module name to allow (typically top-level like "json").
            include_submodules: If True, also allow submodules (e.g., "json.decoder").
        """
        self._temporary_grants.add(module)
        if include_submodules:
            self._temporary_submodule_grants.add(module)
        _log.debug(
            "Temporary import grant added: %s (submodules=%s)", module, include_submodules
        )

    def clear_temporary_grants(self) -> None:
        """Clear all temporary grants."""
        self._temporary_grants.clear()
        self._temporary_submodule_grants.clear()
        _log.debug("Temporary import grants cleared")

    def reload(self, config: ImportConfig | None) -> None:
        """Reload import whitelist from a new config.

        Args:
            config: New import configuration.
        """
        from activecontext.config.schema import ImportConfig

        if config is None:
            config = ImportConfig()

        self.allowed_modules = set(config.allowed_modules)
        self.allow_submodules = config.allow_submodules
        self.allow_all = config.allow_all
        _log.debug("Import whitelist reloaded: %d modules", len(self.allowed_modules))


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


def write_import_to_config(
    cwd: Path, module: str, include_submodules: bool = False
) -> None:
    """Append an import permission to .ac/config.yaml.

    Creates the config file and directory if they don't exist.
    If the file exists, the new module is appended to sandbox.imports.allowed_modules.

    Args:
        cwd: Working directory (project root).
        module: Module name to allow (typically top-level like "json").
        include_submodules: If True, also set allow_submodules to True.
    """
    config_path = cwd / ".ac" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Ensure sandbox.imports exists
    data.setdefault("sandbox", {})
    data["sandbox"].setdefault("imports", {})
    data["sandbox"]["imports"].setdefault("allowed_modules", [])

    # Check if module already exists
    existing_modules = set(data["sandbox"]["imports"]["allowed_modules"])
    if module in existing_modules:
        _log.debug("Import module already allowed: %s", module)
        # Still might need to update allow_submodules
        if include_submodules:
            data["sandbox"]["imports"]["allow_submodules"] = True
    else:
        # Add new module
        data["sandbox"]["imports"]["allowed_modules"].append(module)

    # Update allow_submodules if needed
    if include_submodules:
        data["sandbox"]["imports"]["allow_submodules"] = True

    # Write back with proper YAML formatting
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    _log.info(
        "Added import permission to %s: %s (submodules=%s)",
        config_path,
        module,
        include_submodules,
    )


# =============================================================================
# Website Permission System
# =============================================================================


@dataclass
class WebsitePermissionDenied(Exception):
    """Raised when website access is denied.

    This exception is caught by Timeline to trigger ACP permission requests.
    """

    url: str  # The URL that was denied
    method: str  # HTTP method (GET, POST, etc.)
    parsed_url: dict[str, str] | None = None  # Parsed URL components

    def __str__(self) -> str:
        return f"Website access denied: {self.method} {self.url}"


@dataclass
class WebsitePermissionRule:
    """A website permission rule."""

    pattern: str  # URL pattern (glob or typed placeholders)
    methods: set[str]  # Allowed HTTP methods (e.g., {"GET", "POST"})
    allow: bool  # True to allow, False to deny


class URLTypeValidator:
    """Validates captured values for URL-specific types."""

    @staticmethod
    def validate(value: str, type_name: str) -> bool:
        """Validate value against type.

        Args:
            value: Captured value to validate.
            type_name: Type name (domain, host, url, etc.).

        Returns:
            True if value is valid for the type.
        """
        match type_name:
            case "domain":
                return URLTypeValidator._is_valid_domain(value)
            case "host":
                return URLTypeValidator._is_valid_host(value)
            case "url":
                return URLTypeValidator._is_valid_url(value)
            case "subdomain":
                return URLTypeValidator._is_valid_subdomain(value)
            case "protocol":
                return value.lower() in ("http", "https", "ws", "wss")
            case "port":
                return URLTypeValidator._is_valid_port(value)
            case "path" | "endpoint" | "str":
                return True  # Any string accepted
            case "int":
                try:
                    int(value)
                    return True
                except ValueError:
                    return False
            case _:
                return True

    @staticmethod
    def _is_valid_domain(value: str) -> bool:
        """Check if value is a valid domain name."""
        # Domain: letters, digits, hyphens, dots
        # Must not start/end with hyphen or dot
        pattern = r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$"
        return bool(re.match(pattern, value))

    @staticmethod
    def _is_valid_host(value: str) -> bool:
        """Check if value is a valid hostname (domain, IPv4, or IPv6)."""
        # Try domain
        if URLTypeValidator._is_valid_domain(value):
            return True

        # Try IPv4
        if URLTypeValidator._is_valid_ipv4(value):
            return True

        # Try IPv6 (with or without brackets)
        value_stripped = value.strip("[]")
        if URLTypeValidator._is_valid_ipv6(value_stripped):
            return True

        return False

    @staticmethod
    def _is_valid_ipv4(value: str) -> bool:
        """Check if value is a valid IPv4 address."""
        import socket

        try:
            socket.inet_pton(socket.AF_INET, value)
            return True
        except (socket.error, OSError):
            return False

    @staticmethod
    def _is_valid_ipv6(value: str) -> bool:
        """Check if value is a valid IPv6 address."""
        import socket

        try:
            socket.inet_pton(socket.AF_INET6, value)
            return True
        except (socket.error, OSError):
            return False

    @staticmethod
    def _is_valid_url(value: str) -> bool:
        """Check if value is a valid URL."""
        from urllib.parse import urlparse

        try:
            result = urlparse(value)
            # Must have scheme and netloc
            return bool(result.scheme and result.netloc)
        except Exception:
            return False

    @staticmethod
    def _is_valid_subdomain(value: str) -> bool:
        """Check if value is a valid subdomain (single label)."""
        # Single label: alphanumeric, hyphens (not at start/end)
        pattern = r"^[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$"
        return bool(re.match(pattern, value))

    @staticmethod
    def _is_valid_port(value: str) -> bool:
        """Check if value is a valid port number."""
        try:
            port = int(value)
            return 1 <= port <= 65535
        except ValueError:
            return False

    @staticmethod
    def get_regex_pattern(type_name: str) -> str:
        """Get regex pattern for a type.

        Args:
            type_name: Type name.

        Returns:
            Regex pattern string.
        """
        match type_name:
            case "domain":
                # Domain: alphanumeric with dots and hyphens
                return r"[a-zA-Z0-9\-\.]+"
            case "host":
                # Host: domain or IP (permissive pattern)
                return r"[a-zA-Z0-9\-\.\[\]::]+"
            case "url":
                # URL: protocol + host + optional path/query/fragment
                return r"[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s]+"
            case "subdomain":
                return r"[a-zA-Z0-9\-]+"
            case "path" | "endpoint":
                return r"[^?#]*"
            case "protocol":
                return r"https?"
            case "port":
                return r"\d+"
            case "int":
                return r"\d+"
            case "str":
                return r"[^/]*"
            case _:
                return r".+"


class URLPatternMatcher:
    """Matches URLs against patterns with typed placeholders."""

    def __init__(self) -> None:
        self._validator = URLTypeValidator()

    def match(self, pattern: str, url: str) -> MatchResult:
        """Match URL against pattern.

        Args:
            pattern: Pattern with optional typed placeholders.
            url: URL to match.

        Returns:
            MatchResult with matched status and captures.
        """
        # Check if pattern uses typed placeholders
        if "{" in pattern and ":" in pattern:
            return self._match_typed(pattern, url)
        else:
            # Legacy fnmatch glob
            if fnmatch.fnmatch(url, pattern):
                return MatchResult(matched=True)
            return MatchResult(matched=False)

    def _match_typed(self, pattern: str, url: str) -> MatchResult:
        """Match using typed placeholders.

        Args:
            pattern: Pattern with typed placeholders.
            url: URL to match.

        Returns:
            MatchResult with captures.
        """
        captures: dict[str, str] = {}

        # Find all placeholders in pattern
        placeholder_pattern = re.compile(
            r"\{([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)\}"
        )
        placeholders: list[tuple[str, str]] = []  # (name, type)

        # Build regex pattern by replacing placeholders
        regex_pattern = re.escape(pattern)

        for match in placeholder_pattern.finditer(pattern):
            full_match = re.escape(match.group(0))
            name = match.group(1)
            type_name = match.group(2)
            placeholders.append((name, type_name))

            # Replace escaped placeholder with regex group
            type_regex = self._validator.get_regex_pattern(type_name)
            regex_pattern = regex_pattern.replace(full_match, f"({type_regex})", 1)

        # Try to match
        try:
            regex_match = re.fullmatch(regex_pattern, url)
            if not regex_match:
                return MatchResult(matched=False)

            # Extract and validate captures
            for i, (name, type_name) in enumerate(placeholders):
                value = regex_match.group(i + 1)
                if not self._validator.validate(value, type_name):
                    return MatchResult(matched=False)
                captures[name] = value

            return MatchResult(matched=True, captures=captures)
        except Exception:
            return MatchResult(matched=False)


@dataclass
class WebsitePermissionManager:
    """Manages website access permissions."""

    rules: list[WebsitePermissionRule] = field(default_factory=list)
    deny_by_default: bool = True
    allow_localhost: bool = False
    cwd: Path | None = None
    _temporary_grants: set[tuple[str, str]] = field(default_factory=set)
    _url_matcher: URLPatternMatcher = field(default_factory=URLPatternMatcher)

    @classmethod
    def from_config(
        cls, config: SandboxConfig | None, cwd: Path | None = None
    ) -> WebsitePermissionManager:
        """Create from config."""
        if config is None:
            return cls(deny_by_default=True, allow_localhost=False, cwd=cwd)

        rules = [
            WebsitePermissionRule(
                pattern=p.pattern,
                methods={m.upper() for m in p.methods},
                allow=p.allow,
            )
            for p in config.website_permissions
        ]

        return cls(
            rules=rules,
            deny_by_default=config.website_deny_by_default,
            allow_localhost=config.allow_localhost,
            cwd=cwd,
        )

    def check_access(self, url: str, method: str = "GET") -> bool:
        """Check if access is permitted."""
        method = method.upper()

        # Check temporary grants
        if (url, method) in self._temporary_grants or (url, "ALL") in self._temporary_grants:
            return True

        # Auto-grant localhost if configured
        if self.allow_localhost and self._is_localhost(url):
            return True

        # Check rules (first match wins)
        for rule in self.rules:
            result = self._url_matcher.match(rule.pattern, url)
            if result.matched:
                if "ALL" in rule.methods or method in rule.methods:
                    return rule.allow

        # No match - use default policy
        return not self.deny_by_default

    def _is_localhost(self, url: str) -> bool:
        """Check if URL is localhost."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            hostname = (parsed.hostname or "").lower()
            return hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
        except Exception:
            return False

    def grant_temporary(self, url: str, method: str = "GET") -> None:
        """Grant temporary access (allow once)."""
        self._temporary_grants.add((url, method.upper()))

    def clear_temporary_grants(self) -> None:
        """Clear all temporary grants."""
        self._temporary_grants.clear()

    def reload(self, config: SandboxConfig | None) -> None:
        """Reload from new config."""
        new_manager = WebsitePermissionManager.from_config(config, self.cwd)
        self.rules = new_manager.rules
        self.deny_by_default = new_manager.deny_by_default
        self.allow_localhost = new_manager.allow_localhost

    def list_permissions(self) -> list[dict[str, Any]]:
        """List all rules."""
        return [
            {"pattern": r.pattern, "methods": sorted(r.methods), "allow": r.allow}
            for r in self.rules
        ]


def make_safe_fetch(website_permission_manager: WebsitePermissionManager) -> Any:
    """Create permission-checked fetch() wrapper."""

    async def safe_fetch(
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: Any = None,
        json: Any = None,
        timeout: float = 30.0,
    ) -> Any:
        # Check permission
        if not website_permission_manager.check_access(url, method):
            from urllib.parse import urlparse

            parsed = urlparse(url)
            raise WebsitePermissionDenied(
                url=url,
                method=method,
                parsed_url={
                    "scheme": parsed.scheme,
                    "netloc": parsed.netloc,
                    "path": parsed.path,
                },
            )

        # Execute request
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(
                method=method,
                url=url,
                headers=headers,
                data=data,
                json=json,
            )

    return safe_fetch


def write_website_permission_to_config(
    cwd: Path, url: str, method: str = "GET"
) -> None:
    """Write permission to .ac/config.yaml."""
    config_path = cwd / ".ac" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Ensure structure exists
    data.setdefault("sandbox", {})
    data["sandbox"].setdefault("website_permissions", [])

    # Check if pattern exists - merge methods
    method_upper = method.upper()
    pattern_exists = False
    for p in data["sandbox"]["website_permissions"]:
        if isinstance(p, dict) and p.get("pattern") == url:
            methods = [m.upper() for m in p.get("methods", [])]
            if method_upper not in methods:
                methods.append(method_upper)
                p["methods"] = methods
            pattern_exists = True
            break

    # Add new rule if pattern doesn't exist
    if not pattern_exists:
        data["sandbox"]["website_permissions"].append(
            {
                "pattern": url,
                "methods": [method_upper],
            }
        )

    # Write back
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    _log.info("Added website permission to %s: %s (%s)", config_path, url, method)
