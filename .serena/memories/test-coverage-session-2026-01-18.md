# Test Coverage Audit - Updated 2026-01-20

## Current State

**699 tests passing, 61% coverage (8,049 statements, 3,156 missing)**

## Critical Coverage Gaps

### Zero Coverage (478 statements)
| Module | Statements |
|--------|------------|
| `dashboard/*` (all files) | 447 |
| `clean.py` | 31 |

### Very Low Coverage (<30%)
| Module | Coverage | Missing |
|--------|----------|---------|
| `mcp/transport.py` | 16% | 27 |
| `transport/acp/agent.py` | 21% | 512 |
| `mcp/client.py` | 27% | 131 |

### Low Coverage (30-50%)
| Module | Coverage | Missing |
|--------|----------|---------|
| `agents/handle.py` | 30% | 49 |
| `agents/manager.py` | 31% | 55 |
| `context/view.py` | 31% | 97 |
| `agents/registry.py` | 33% | 28 |
| `watching/watcher.py` | 49% | 70 |
| `session/timeline.py` | 49% | 540 |

### Moderate Coverage (50-70%)
| Module | Coverage | Missing |
|--------|----------|---------|
| `mcp/permissions.py` | 53% | 25 |
| `context/content.py` | 54% | 30 |
| `session/storage.py` | 60% | 40 |
| `coordination/scratchpad.py` | 62% | 90 |
| `session/session_manager.py` | 64% | 200 |
| `context/nodes.py` | 65% | 444 |

## Priority Actions

### P0 - Critical
1. `session/timeline.py` - Core execution engine (540 stmts missing)
2. `transport/acp/agent.py` - ACP protocol (512 stmts missing)

### P1 - High Impact
1. `context/nodes.py` - Node types (444 stmts missing)
2. `mcp/client.py` - MCP client (131 stmts missing)
3. `agents/*` - Agent subsystem (132 stmts total)

### Quick Wins
1. `clean.py` - 31 stmts, simple logic
2. `mcp/transport.py` - 27 stmts
3. `agents/registry.py` - 28 stmts

## Target: 80% coverage

## Commands
```bash
uv run pytest                           # Run all tests
uv run pytest --cov=src/activecontext   # Run with coverage
uv run pytest tests/test_file.py -v     # Run specific file
```