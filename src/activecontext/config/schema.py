"""Configuration schema dataclasses for ActiveContext.

Defines the structure of configuration at all levels (system, user, project).
All fields are optional to support partial configs that merge together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class RoleProviderConfig:
    """User's saved provider/model choice for a role.

    Allows persisting the user's preferred provider and optional model override per role.
    """

    role: str  # e.g., "coding", "fast", "reasoning"
    provider: str  # e.g., "anthropic", "openai"
    model: str | None = None  # Optional model override (e.g., "gpt-5-mini")


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    role: str | None = None  # Last selected role (e.g., "coding")
    provider: str | None = None  # Last selected provider (e.g., "anthropic")
    role_providers: list[RoleProviderConfig] = field(default_factory=list)  # Per-role prefs
    api_base: str | None = None  # Custom endpoint
    max_tokens: int | None = None  # Default: 4096


@dataclass
class SessionModeConfig:
    """A session mode definition."""

    id: str
    name: str
    description: str = ""



@dataclass
class StartupConfig:
    """Configuration for session startup scripts.

    Startup statements are DSL statements executed on NEW session creation only.
    When loading an existing session, the context graph is restored directly.

    Example config.yaml:
        session:
          startup:
            statements:
              - 'readme = view("README.md", tokens=2000)'
              - 'mcp_connect("filesystem")'
            skip_default_context: false
    """

    statements: list[str] = field(default_factory=list)  # DSL statements to execute
    skip_default_context: bool = False  # Skip loading CONTEXT_GUIDE.md


@dataclass
class FileWatchConfig:
    """File watching configuration.

    Controls how the agent monitors files for external changes.
    Changes to watched files can wake the agent or be queued for batch processing.

    Example config.yaml:
        session:
          file_watch:
            enabled: true
            poll_interval: 1.0
            watch_project: false
            project_patterns:
              - "**/*.py"
              - "**/*.md"
            ignore_patterns:
              - "**/__pycache__/**"
              - "**/.git/**"
    """

    enabled: bool = True  # Enable file watching
    poll_interval: float = 1.0  # Seconds between polling cycles
    watch_project: bool = False  # Watch all files under project root
    project_patterns: list[str] = field(
        default_factory=lambda: ["**/*.py", "**/*.md", "**/*.yaml"]
    )  # Glob patterns for project watching
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            "**/__pycache__/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/*.pyc",
        ]
    )  # Patterns to ignore
    max_files: int = 1000  # Maximum files to watch (prevents memory issues)


@dataclass
class SessionConfig:
    """Session defaults configuration."""

    modes: list[SessionModeConfig] = field(default_factory=list)
    default_mode: str | None = None  # Default: "normal"
    startup: StartupConfig = field(default_factory=StartupConfig)
    file_watch: FileWatchConfig = field(default_factory=FileWatchConfig)


@dataclass
class ProjectionConfig:
    """Projection engine configuration."""

    pass  # Budget removed - agent manages via node visibility and line ranges  # Default: 8000  # Default: 0.4  # Default: 0.2


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
class UserConfig:
    """User identity configuration.

    Configures the display name shown in conversation rendering.
    Falls back to USER/USERNAME environment variable if not set.
    """

    display_name: str | None = None  # e.g., "Ace"


class MCPConnectMode(Enum):
    """Connection mode for MCP servers.

    - CRITICAL: Must connect successfully on startup, fail session if cannot
    - AUTO: Auto-connect on startup, warn but continue if fails
    - MANUAL: Only connect when explicitly called via mcp_connect()
    - NEVER: Disabled, cannot be connected
    """

    CRITICAL = "critical"
    AUTO = "auto"
    MANUAL = "manual"
    NEVER = "never"


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server.

    Supports three transport types:
        - stdio: Spawns a subprocess (requires command)
        - streamable-http: Connects to HTTP endpoint (requires url)
        - sse: Connects to SSE endpoint (requires url)
    """

    name: str  # Unique identifier (e.g., "filesystem", "github")
    command: list[str] | None = None  # For stdio: ["npx", "-y", "@mcp/server"]
    args: list[str] = field(default_factory=list)  # Additional command args
    env: dict[str, str] = field(default_factory=dict)  # Environment vars (supports ${VAR})
    url: str | None = None  # For streamable-http/sse: "http://localhost:8000/sse"
    headers: dict[str, str] = field(default_factory=dict)  # HTTP headers for sse/http
    transport: str = "stdio"  # "stdio", "streamable-http", or "sse"
    connect: MCPConnectMode = MCPConnectMode.MANUAL  # Connection mode
    timeout: float = 30.0  # Connection timeout in seconds


@dataclass
class MCPConfig:
    """MCP client configuration.

    Example config.yaml:
        mcp:
          allow_dynamic_servers: true
          servers:
            - name: filesystem
              command: ["npx", "-y", "@modelcontextprotocol/server-filesystem"]
              args: ["/home/user/allowed"]
              auto_connect: true
            - name: github
              command: ["npx", "-y", "@modelcontextprotocol/server-github"]
              env:
                GITHUB_TOKEN: "${GITHUB_TOKEN}"
    """

    servers: list[MCPServerConfig] = field(default_factory=list)
    allow_dynamic_servers: bool = True  # Allow mcp_connect() with inline config



@dataclass
class ACPConfig:
    """ACP transport configuration.

    Controls behavior of the ACP (Agent-Client Protocol) transport layer.

    Attributes:
        out_of_band_update: If True, send session updates immediately via
            notifications (async model). If False, queue updates that occur
            between prompts and flush them when the next prompt arrives
            (sync model). Default False for compatibility with clients like
            Rider that expect responses after PromptResponse.
    """

    out_of_band_update: bool = False

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
    user: UserConfig = field(default_factory=UserConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    acp: ACPConfig = field(default_factory=ACPConfig)

    # Extension point for future config sections
    extra: dict[str, Any] = field(default_factory=dict)
