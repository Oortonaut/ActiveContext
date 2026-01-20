# Test Coverage Improvement Session - 2026-01-18

## Completed Work

### 1. Fixed Test Hanging Issue
**Root cause**: `Timeline` creates background `asyncio.Task`s for shell commands but had no cleanup method. When pytest-asyncio closes the event loop, lingering tasks cause hangs on Windows IOCP.

**Fix applied** (commit `e53be49`):
- Added `Timeline.close()` method in `src/activecontext/session/timeline.py`
- Added `__aenter__`/`__aexit__` for `async with` support
- Updated 18 tests in `tests/test_permissions.py` with `try/finally` cleanup

### 2. Fixed Secret Loading Priority
**Issue**: Tests using `monkeypatch.delenv()` couldn't clear API keys because `fetch_secret()` checked `.env` file before `os.environ`.

**Fix applied**:
- Changed `fetch_secret()` in `src/activecontext/config/secrets.py` to check `os.environ` first
- Renamed `.env` to `.env.secrets` for API keys
- Updated `.gitignore` to include `.env.secrets`

### 3. Added Test Dependencies
- `pytest-cov` for coverage reporting
- `pytest-timeout` for test timeouts

## Current State

**All 612 tests pass in ~8 seconds with 59% coverage**

## Coverage Gaps to Address

### Zero Coverage Modules (Priority)
| Module | Statements | Purpose |
|--------|-----------|---------|
| `dashboard/` (all files) | 381 | Web monitoring interface |
| `clean.py` | 31 | Build artifact cleanup |

### Low Coverage Modules (High Impact)
| Module | Coverage | Missing |
|--------|----------|---------|
| `transport/acp/agent.py` | 19% | 374 stmts - ACP transport adapter |
| `mcp/client.py` | 27% | 131 stmts - MCP client |
| `context/view.py` | 31% | 97 stmts - File view node |
| `session/storage.py` | 37% | 64 stmts - Session persistence |
| `session/timeline.py` | 40% | 570 stmts - Core execution engine |

### Moderate Coverage (Could Improve)
| Module | Coverage |
|--------|----------|
| `agents/handle.py` | 30% |
| `agents/manager.py` | 31% |
| `agents/registry.py` | 33% |
| `mcp/permissions.py` | 53% |
| `config/secrets.py` | 54% |
| `context/content.py` | 54% |

## Recommended Next Steps

1. **Add integration/crosscutting tests** for:
   - Full session lifecycle (create → execute → close)
   - Multi-agent coordination
   - MCP client operations
   - Dashboard routes (if keeping dashboard)

2. **Increase timeline.py coverage** - Core execution engine at 40%:
   - Statement replay/rollback
   - Error handling paths
   - More shell/lock scenarios

3. **Add ACP transport tests** - Currently 19%:
   - Agent initialization with various configs
   - Session management
   - Update handling

4. **Consider removing dashboard** if not needed (0% coverage, 381 statements)

## Commands Reference
```bash
uv run pytest                           # Run all tests
uv run pytest --cov=src/activecontext   # Run with coverage
uv run pytest tests/test_file.py -v     # Run specific file
```
