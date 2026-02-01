# JetBrains Rider Terminal Quirks and Workarounds

This document catalogs known issues and workarounds when running ActiveContext in the JetBrains Rider terminal environment.

## Terminal Lifecycle Issues

### Issue: Subprocess Hang on Chat Deletion (Active Chat)

**Symptom**: When deleting the *currently active* chat in Rider, the ACP agent process hangs instead of terminating cleanly.

**Root Cause**: Rider bug where stdio pipes are not properly closed when deleting the active chat. The subprocess continues waiting for input that will never arrive.

**Workaround**: Delete chats from the chat list sidebar instead of while the chat is active. Deleting from the list correctly closes pipes and terminates the subprocess.

**Rider Versions Affected**: Rider 2025.3 (confirmed), likely earlier versions

**Status**: Reported to JetBrains

**Expected Behavior**: When properly closing a chat:
1. Rider closes stdio pipes
2. Agent detects EOF on stdin
3. Agent exits gracefully
4. Rider cleans up subprocess resources

### Issue: ANSI Color Code Support

**Symptom**: ANSI color codes may not render correctly in older Rider versions.

**Detection**: Check `TERM_PROGRAM` environment variable for "jetbrains" or "rider".

**Workaround**:
```python
from activecontext.terminal.capabilities import get_capabilities

caps = get_capabilities()
if caps.is_rider:
    # Use conservative color support
    use_color = caps.color_support != ColorSupport.NONE
```

**Rider Versions Affected**: Pre-2024.1 versions have limited ANSI support

**Status**: Improved in recent Rider versions

## PTY and Interactive Commands

### Issue: PTY Not Available in Rider Terminal

**Symptom**: Interactive commands that require a PTY (ssh, password prompts) fail or behave incorrectly.

**Root Cause**: Rider's embedded terminal does not expose PTY capabilities to subprocess spawned from ACP agents.

**Workaround**: Use Rider's built-in terminal for interactive commands, or spawn commands via Windows Terminal:
```python
# For interactive commands, use Rider's shell integration
if caps.is_rider:
    # Direct user to run interactively in Rider terminal
    return "Please run this command in the Rider terminal window"
```

**Alternative**: Use ACP terminal delegation (if available) to run commands in the IDE's terminal pane.

### Issue: Stdin Redirection Not Supported

**Symptom**: Commands expecting stdin input hang or fail.

**Root Cause**: ACP stdio transport uses stdin for JSON-RPC messages, not user input.

**Workaround**: For commands requiring input:
1. Pre-provide input via command-line args or files
2. Use expect-style automation (pexpect on Unix)
3. Delegate to IDE terminal via ACP `terminal/execute` if available

## Terminal Capability Detection

### Issue: TERM Variable Not Set

**Symptom**: `os.environ.get("TERM")` returns None or empty string in Rider.

**Detection**:
```python
term_program = os.environ.get("TERM_PROGRAM")
if term_program and "jetbrains" in term_program.lower():
    # Running in JetBrains IDE
    is_rider = True
```

**Workaround**: Use fallback detection based on `TERM_PROGRAM`:
- Set `TERM_PROGRAM=rider` in ACP launch config
- Detect via process tree inspection (parent process name)

### Issue: Unicode Support Detection

**Symptom**: Unicode characters may render as `?` or box characters even when terminal supports UTF-8.

**Detection**: Rider's terminal uses UTF-8 by default in recent versions, but older versions may not.

**Workaround**:
```python
if caps.is_rider:
    # Check Rider version
    version = os.environ.get("TERM_PROGRAM_VERSION", "")
    if version < "2024.1":
        # Conservative: assume no unicode
        caps.unicode_support = False
```

## Output Capture and Display

### Issue: Output Truncation

**Symptom**: Long command output may be truncated or lost.

**Root Cause**: Rider terminal has buffer limits for subprocess output.

**Workaround**:
- Set reasonable `output_limit` on shell commands (default 50KB)
- For large outputs, redirect to file and read incrementally
- Use streaming output when possible

### Issue: Progress Indicators Not Updating

**Symptom**: Progress bars and spinners from commands don't update in real-time.

**Root Cause**: Buffering in subprocess pipes, or lack of PTY (commands detect non-TTY and disable fancy output).

**Workaround**:
- Commands detect non-interactive environment
- Use `--no-progress` or similar flags for cleaner output
- Poll for status instead of relying on progress bars

## Environment Variable Handling

### Issue: PATH Not Inherited Correctly

**Symptom**: Commands fail with "not found" even though they're in PATH.

**Root Cause**: Rider's terminal environment may differ from system shell environment.

**Workaround**:
```python
# Explicitly set PATH from Rider's environment
import os
process_env = os.environ.copy()

# Add common binary locations
if sys.platform == "win32":
    # Ensure user bin directories are in PATH
    user_bin = Path.home() / "bin"
    if user_bin.exists():
        process_env["PATH"] = str(user_bin) + os.pathsep + process_env.get("PATH", "")
```

### Issue: Working Directory Mismatch

**Symptom**: Commands execute in wrong directory.

**Root Cause**: ACP agent's working directory may not match Rider project root.

**Workaround**:
- Always specify `cwd` explicitly in shell commands
- Use ACP context to get project root
- Validate paths before execution

## Performance Considerations

### Issue: High Latency for Subprocess Spawn

**Symptom**: Shell commands have noticeable startup delay (100-500ms).

**Root Cause**: Windows subprocess creation overhead + Rider IDE monitoring.

**Mitigation**:
- Batch commands when possible
- Use shell sessions for multiple commands
- Cache command results when appropriate

### Issue: Memory Usage Growth

**Symptom**: Agent memory usage grows with each command execution.

**Root Cause**: Output buffers, subprocess handles, or event loop references not cleaned up.

**Mitigation**:
- Explicitly close subprocess handles
- Limit output capture size
- Periodic garbage collection for long-running agents

## Logging and Debugging

### Recommended Environment Variables

For debugging terminal issues in Rider:

```json
{
  "agent_servers": {
    "activecontext": {
      "command": "python",
      "args": ["-m", "activecontext"],
      "env": {
        "AC_LOG": "C:\\Users\\You\\activecontext.log",
        "AC_LOG_CONTEXT": "C:\\Users\\You\\ctx-dumps",
        "TERM_PROGRAM": "rider",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

### Rider Log Locations

- **ACP high-level events**: `%LOCALAPPDATA%\JetBrains\Rider2025.3\log\acp\acp.log`
- **Raw JSON-RPC messages**: `%LOCALAPPDATA%\JetBrains\Rider2025.3\log\acp\acp-transport.log`
- **Agent logs** (if AC_LOG set): User-specified path

## Best Practices

### Detection Pattern

```python
from activecontext.terminal.capabilities import get_capabilities

def is_rider_environment() -> bool:
    """Detect if running in Rider terminal."""
    caps = get_capabilities()
    return caps.is_rider or (
        caps.term_program and "jetbrains" in caps.term_program.lower()
    )

def get_shell_executor():
    """Get appropriate shell executor for environment."""
    if is_rider_environment():
        # Use conservative settings for Rider
        return SubprocessTerminalExecutor(
            default_cwd=".",
            # Disable PTY (not supported)
            # Use shorter timeouts (Rider may have limits)
            # Smaller output buffers
        )
    else:
        # Full capabilities for other environments
        return SubprocessTerminalExecutor(default_cwd=".")
```

### Graceful Degradation

```python
def format_output(text: str) -> str:
    """Format output with graceful degradation."""
    caps = get_capabilities()

    if caps.is_rider:
        # Conservative formatting for Rider
        if caps.supports_color:
            # Use basic 16-color ANSI
            return format_with_fallback(text, color="green")
        else:
            # Plain text only
            return text
    else:
        # Full formatting for other terminals
        return format_with_fallback(text, color="green", bold=True)
```

## Future Improvements

- **ACP Terminal Integration**: Use Rider's terminal pane directly via ACP protocol extensions
- **ConPTY Support**: Windows Pseudo Console API for interactive commands
- **Streaming Output**: Real-time output updates via ACP notifications
- **Environment Sync**: Automatic PATH and environment variable inheritance from Rider

## References

- [ACP Protocol Specification](../../docs/acp-protocol.md)
- [Rider ACP Documentation](https://www.jetbrains.com/help/rider/acp.html)
- [Windows Console APIs](https://docs.microsoft.com/en-us/windows/console/)
