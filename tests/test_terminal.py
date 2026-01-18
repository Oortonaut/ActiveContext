"""Tests for terminal execution functionality."""

import asyncio
import sys

import pytest

from activecontext.terminal.result import ShellResult
from activecontext.terminal.subprocess_executor import SubprocessTerminalExecutor
from activecontext.session.xml_parser import parse_xml_to_python


class TestShellResult:
    """Tests for ShellResult dataclass."""

    def test_success_property(self):
        result = ShellResult(
            command="echo hello",
            exit_code=0,
            output="hello\n",
            truncated=False,
            status="ok",
            signal=None,
            duration_ms=10.0,
        )
        assert result.success is True

    def test_failure_property(self):
        result = ShellResult(
            command="exit 1",
            exit_code=1,
            output="",
            truncated=False,
            status="error",
            signal=None,
            duration_ms=10.0,
        )
        assert result.success is False

    def test_timeout_property(self):
        result = ShellResult(
            command="sleep 100",
            exit_code=None,
            output="Command timed out",
            truncated=False,
            status="timeout",
            signal="SIGKILL",
            duration_ms=1000.0,
        )
        assert result.success is False

    def test_repr_ok(self):
        result = ShellResult(
            command="echo hello",
            exit_code=0,
            output="hello\n",
            truncated=False,
            status="ok",
            signal=None,
            duration_ms=10.0,
        )
        assert "ok" in repr(result)

    def test_repr_error(self):
        result = ShellResult(
            command="exit 1",
            exit_code=1,
            output="",
            truncated=False,
            status="error",
            signal=None,
            duration_ms=10.0,
        )
        assert "error" in repr(result)
        assert "exit=1" in repr(result)


class TestSubprocessTerminalExecutor:
    """Tests for SubprocessTerminalExecutor."""

    @pytest.fixture
    def executor(self, tmp_path):
        return SubprocessTerminalExecutor(default_cwd=str(tmp_path))

    @pytest.mark.asyncio
    async def test_echo_basic(self, executor):
        if sys.platform == "win32":
            result = await executor.execute("cmd", args=["/c", "echo", "hello"])
        else:
            result = await executor.execute("echo", args=["hello"])
        assert result.success
        assert "hello" in result.output
        assert result.status == "ok"
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_echo_multiple_args(self, executor):
        if sys.platform == "win32":
            result = await executor.execute("cmd", args=["/c", "echo", "hello", "world"])
        else:
            result = await executor.execute("echo", args=["hello", "world"])
        assert result.success
        assert "hello" in result.output
        assert "world" in result.output

    @pytest.mark.asyncio
    async def test_command_not_found(self, executor):
        result = await executor.execute("nonexistent_command_xyz")
        assert not result.success
        assert result.exit_code == 127
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_command_failure(self, executor):
        # Use a command that will fail
        if sys.platform == "win32":
            result = await executor.execute("cmd", args=["/c", "exit", "1"])
        else:
            result = await executor.execute("sh", args=["-c", "exit 1"])
        assert not result.success
        assert result.exit_code == 1
        assert result.status == "error"

    @pytest.mark.asyncio
    async def test_timeout(self, executor):
        # Use a command that takes longer than timeout
        if sys.platform == "win32":
            result = await executor.execute("ping", args=["-n", "10", "127.0.0.1"], timeout=0.1)
        else:
            result = await executor.execute("sleep", args=["10"], timeout=0.1)
        assert not result.success
        assert result.status == "timeout"
        assert result.exit_code is None
        assert result.signal == "SIGKILL"

    @pytest.mark.asyncio
    async def test_custom_cwd(self, executor, tmp_path):
        # Create a test directory and file
        test_dir = tmp_path / "subdir"
        test_dir.mkdir()
        (test_dir / "test.txt").write_text("hello")

        if sys.platform == "win32":
            result = await executor.execute("cmd", args=["/c", "dir"], cwd=str(test_dir))
        else:
            result = await executor.execute("ls", cwd=str(test_dir))

        assert result.success
        assert "test.txt" in result.output

    @pytest.mark.asyncio
    async def test_output_truncation(self, executor):
        # Generate a lot of output
        if sys.platform == "win32":
            cmd, args = "cmd", ["/c", "for /L %i in (1,1,1000) do @echo line%i"]
        else:
            cmd, args = "sh", ["-c", "for i in $(seq 1 1000); do echo line$i; done"]

        result = await executor.execute(cmd, args=args, output_limit=100)
        assert result.truncated
        assert "truncated" in result.output

    @pytest.mark.asyncio
    async def test_env_variables(self, executor):
        if sys.platform == "win32":
            result = await executor.execute(
                "cmd", args=["/c", "echo %TEST_VAR%"],
                env={"TEST_VAR": "test_value"}
            )
        else:
            result = await executor.execute(
                "sh", args=["-c", "echo $TEST_VAR"],
                env={"TEST_VAR": "test_value"}
            )
        assert result.success
        assert "test_value" in result.output


class TestXmlParserShell:
    """Tests for XML parser shell tag support."""

    def test_shell_basic(self):
        xml = '<shell command="echo" args="hello"/>'
        py = parse_xml_to_python(xml)
        assert "shell('echo'" in py
        assert "args=['hello']" in py

    def test_shell_multiple_args(self):
        xml = '<shell command="pytest" args="tests/,-v,--tb=short"/>'
        py = parse_xml_to_python(xml)
        assert "shell('pytest'" in py
        assert "args=['tests/', '-v', '--tb=short']" in py

    def test_shell_with_timeout(self):
        xml = '<shell command="sleep" args="10" timeout="5"/>'
        py = parse_xml_to_python(xml)
        assert "shell('sleep'" in py
        assert "timeout=5" in py

    def test_shell_no_args(self):
        xml = '<shell command="pwd"/>'
        py = parse_xml_to_python(xml)
        assert py == "shell('pwd')"

    def test_shell_requires_command(self):
        xml = '<shell args="hello"/>'
        with pytest.raises(ValueError, match="requires 'command' attribute"):
            parse_xml_to_python(xml)


class TestTimelineShellIntegration:
    """Tests for shell() function in Timeline namespace."""

    @pytest.fixture
    async def timeline(self, tmp_path):
        """Create timeline and cleanup after test."""
        from activecontext.session.timeline import Timeline
        tl = Timeline(session_id="test", cwd=str(tmp_path))
        yield tl
        # Cancel all background shell tasks
        for task in tl._shell_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_shell_in_timeline(self, timeline):
        import asyncio

        # Execute shell via timeline (Windows needs cmd.exe for echo)
        if sys.platform == "win32":
            result = await timeline.execute_statement('result = shell("cmd", args=["/c", "echo", "hello"])')
        else:
            result = await timeline.execute_statement('result = shell("echo", args=["hello"])')
        assert result.status.value == "ok"

        # Wait for background task and process results
        await asyncio.sleep(0.5)
        timeline.process_pending_shell_results()

        # Check result is in namespace
        ns = timeline.get_namespace()
        assert "result" in ns
        shell_result = ns["result"]
        assert shell_result.is_success
        assert "hello" in shell_result.output

    @pytest.mark.asyncio
    async def test_shell_as_expression(self, timeline):
        import asyncio

        # Execute shell as expression (result printed)
        if sys.platform == "win32":
            result = await timeline.execute_statement('shell("cmd", args=["/c", "echo", "test"])')
        else:
            result = await timeline.execute_statement('shell("echo", args=["test"])')
        assert result.status.value == "ok"
        # The ShellNode repr should be in stdout
        assert "ShellNode" in result.stdout

    @pytest.mark.asyncio
    async def test_shell_command_not_found(self, timeline):
        import asyncio

        result = await timeline.execute_statement(
            'r = shell("nonexistent_command_xyz_123")'
        )
        assert result.status.value == "ok"  # Statement executed successfully

        # Wait for background task and process results
        await asyncio.sleep(0.5)
        timeline.process_pending_shell_results()

        ns = timeline.get_namespace()
        assert "r" in ns
        assert not ns["r"].is_success
        assert ns["r"].exit_code == 127
