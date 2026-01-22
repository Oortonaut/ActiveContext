# ActiveContext

Agent loop architecture where an LLM controls a structured, reversible working context through a Python statement timeline.

## Features

- **Executable context**: LLM executes Python statements that manipulate context objects (views, groups, shells, MCP connections)
- **Tick-driven updates**: All automatic mutations occur at deterministic tick boundaries
- **Reversible timeline**: Re-execute from any statement; checkpoint and restore context state
- **Multiple transports**: Direct Python API or ACP (Agent Client Protocol) for IDE integration
- **MCP integration**: Connect to Model Context Protocol servers for tools and resources
- **Permission system**: Configurable file, shell, import, and web access controls

## Installation

```bash
# Using uv (recommended)
uv add activecontext

# Or with pip
pip install activecontext
```

### Development Installation

```bash
git clone https://github.com/your-org/activecontext.git
cd activecontext
uv sync --all-extras
```

## Quick Start

### Direct Transport (Python API)

```python
from activecontext import ActiveContext

async with ActiveContext() as ctx:
    session = await ctx.create_session(cwd=".")

    # Execute Python to create context objects
    await session.execute('v = text("main.py", tokens=2000)')

    # Stream updates from a prompt
    async for update in session.prompt("Summarize this file"):
        print(update)

    # Access namespace and context objects
    print(session.get_namespace())
```

### ACP Transport (IDE Integration)

Configure in your IDE's `acp.json`:

```json
{
  "agent_servers": {
    "activecontext": {
      "command": "python",
      "args": ["-m", "activecontext"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-..."
      }
    }
  }
}
```

Run directly:

```bash
python -m activecontext
```

## Configuration

ActiveContext uses hierarchical YAML configuration files:

| Level | Windows | Unix |
|-------|---------|------|
| User | `%APPDATA%\activecontext\config.yaml` | `~/.ac/config.yaml` |
| Project | `.ac/config.yaml` | `.ac/config.yaml` |

Example `config.yaml`:

```yaml
llm:
  provider: anthropic
  max_tokens: 8192

projection:
  total_budget: 16000

sandbox:
  allow_cwd: true
  shell_deny_by_default: true
```

## Environment Variables

At least one LLM API key must be set:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude models) |
| `OPENAI_API_KEY` | OpenAI API key (GPT models) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `DEEPSEEK_API_KEY` | DeepSeek API key |

Debug options:

| Variable | Description |
|----------|-------------|
| `ACTIVECONTEXT_LOG` | File path for diagnostic logs |
| `ACTIVECONTEXT_DEBUG` | Print projection contents to stderr |

## Development

```bash
uv run pytest            # Run tests
uv run pytest -xvs       # Verbose, stop on first failure
uv run ruff check src/   # Lint
uv run ruff format src/  # Format
uv run mypy src/         # Type check
```

## Documentation

- [Design Document](DESIGN.md) - Architecture and design decisions
- [ACP Protocol](docs/acp-protocol.md) - Agent Client Protocol specification
- [DSL Reference](src/activecontext/prompts/dsl_reference.md) - Python DSL for context manipulation

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
