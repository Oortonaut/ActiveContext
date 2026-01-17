"""Configuration schema dataclasses for ActiveContext.

Defines the structure of configuration at all levels (system, user, project).
All fields are optional to support partial configs that merge together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoleModelConfig:
    """User's saved model choice for a role.

    Allows persisting the user's preferred model for each role.
    """

    role: str  # e.g., "coding", "fast", "reasoning"
    provider: str  # e.g., "anthropic", "openai"
    model_id: str  # e.g., "claude-sonnet-4-5-20250929"


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str | None = None  # e.g., "anthropic", "openai"
    model: str | None = None  # e.g., "claude-sonnet-4-20250514" (explicit override)
    default_role: str | None = None  # e.g., "coding" - role to use when model not specified
    role_models: list[RoleModelConfig] = field(default_factory=list)  # User's saved choices
    api_key: str | None = None  # Override env var
    api_base: str | None = None  # Custom endpoint
    temperature: float | None = None  # Default: 0.0
    max_tokens: int | None = None  # Default: 4096


@dataclass
class SessionModeConfig:
    """A session mode definition."""

    id: str
    name: str
    description: str = ""


@dataclass
class SessionConfig:
    """Session defaults configuration."""

    modes: list[SessionModeConfig] = field(default_factory=list)
    default_mode: str | None = None  # Default: "normal"


@dataclass
class ProjectionConfig:
    """Projection engine configuration."""

    total_budget: int | None = None  # Default: 8000
    conversation_ratio: float | None = None  # Default: 0.3
    views_ratio: float | None = None  # Default: 0.5
    groups_ratio: float | None = None  # Default: 0.2


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str | None = None  # DEBUG, INFO, WARNING, ERROR
    file: str | None = None  # Log file path


@dataclass
class FilePermissionConfig:
    """A file permission rule for the sandbox.

    Defines a glob pattern and access mode for file operations.
    """

    pattern: str  # Glob pattern (e.g., "*.py", "./data/**")
    mode: str = "read"  # "read", "write", or "all"


@dataclass
class ShellPermissionConfig:
    """A shell permission rule for the sandbox.

    Defines a glob pattern for shell command access control.
    Pattern is matched against the full command string (command + args).
    """

    pattern: str  # Glob pattern (e.g., "git *", "npm run *", "pytest *")
    allow: bool = True  # True to allow, False to deny


@dataclass
class WebsitePermissionConfig:
    """Website permission rule for sandbox.

    Defines URL pattern and allowed HTTP methods for web requests.
    """

    pattern: str  # URL pattern with optional typed placeholders
    methods: list[str] = field(default_factory=lambda: ["GET"])
    allow: bool = True  # True to allow, False to deny


@dataclass
class ImportConfig:
    """Import whitelist configuration for the REPL sandbox.

    Controls which modules can be imported by code executing in the Timeline.
    """

    allowed_modules: list[str] = field(default_factory=list)  # Whitelist of module names
    allow_submodules: bool = True  # Allow submodules of whitelisted modules (e.g., os.path)
    allow_all: bool = False  # Bypass whitelist entirely (insecure, for trusted code)


@dataclass
class SandboxConfig:
    """Sandbox configuration for file access control.

    Controls which files can be read/written by code executing in the Timeline,
    and which shell commands can be executed.
    """

    file_permissions: list[FilePermissionConfig] = field(default_factory=list)
    allow_cwd: bool = True  # Auto-grant read access to cwd
    allow_cwd_write: bool = False  # Auto-grant write access to cwd
    deny_by_default: bool = True  # Deny unlisted paths
    allow_absolute: bool = False  # Allow paths outside cwd
    imports: ImportConfig = field(default_factory=ImportConfig)  # Import whitelist
    shell_permissions: list[ShellPermissionConfig] = field(default_factory=list)
    shell_deny_by_default: bool = True  # Deny unlisted shell commands
    website_permissions: list[WebsitePermissionConfig] = field(default_factory=list)
    website_deny_by_default: bool = True  # Deny unlisted websites
    allow_localhost: bool = False  # Auto-grant access to localhost  # Deny unlisted shell commands


@dataclass
class Config:
    """Root configuration object.

    Aggregates all configuration sections. All fields use default factories
    to ensure partial configs work correctly with deep merging.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    # Extension point for future config sections
    extra: dict[str, Any] = field(default_factory=dict)
