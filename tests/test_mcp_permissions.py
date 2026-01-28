"""Tests for the MCP permission management module."""

from __future__ import annotations

import pytest

from activecontext.mcp.permissions import (
    MCPPermissionDenied,
    MCPPermissionManager,
    MCPPermissionRule,
)

# =============================================================================
# MCPPermissionDenied Tests
# =============================================================================


class TestMCPPermissionDenied:
    """Tests for MCPPermissionDenied exception."""

    def test_init_stores_fields(self):
        """Test exception stores all fields."""
        exc = MCPPermissionDenied(
            server_name="filesystem",
            tool_name="read_file",
            arguments={"path": "/etc/passwd"},
        )

        assert exc.server_name == "filesystem"
        assert exc.tool_name == "read_file"
        assert exc.arguments == {"path": "/etc/passwd"}

    def test_str_format(self):
        """Test exception string format."""
        exc = MCPPermissionDenied(
            server_name="github",
            tool_name="push_code",
            arguments={},
        )

        result = str(exc)

        assert "denied" in result
        assert "github" in result
        assert "push_code" in result

    def test_is_exception(self):
        """Test it can be raised and caught."""
        with pytest.raises(MCPPermissionDenied) as exc_info:
            raise MCPPermissionDenied(
                server_name="test",
                tool_name="tool",
                arguments={},
            )

        assert exc_info.value.server_name == "test"


# =============================================================================
# MCPPermissionRule Tests
# =============================================================================


class TestMCPPermissionRule:
    """Tests for MCPPermissionRule dataclass."""

    def test_default_allow(self):
        """Test default allow is True."""
        rule = MCPPermissionRule(pattern="filesystem.*")

        assert rule.pattern == "filesystem.*"
        assert rule.allow is True

    def test_explicit_deny(self):
        """Test explicit deny rule."""
        rule = MCPPermissionRule(pattern="*.dangerous_*", allow=False)

        assert rule.pattern == "*.dangerous_*"
        assert rule.allow is False


# =============================================================================
# MCPPermissionManager Initialization Tests
# =============================================================================


class TestMCPPermissionManagerInit:
    """Tests for MCPPermissionManager initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        manager = MCPPermissionManager()

        assert manager.rules == []
        assert manager.deny_by_default is True
        assert manager._temporary_grants == set()

    def test_custom_deny_by_default(self):
        """Test custom deny_by_default setting."""
        manager = MCPPermissionManager(deny_by_default=False)

        assert manager.deny_by_default is False

    def test_initial_rules(self):
        """Test initialization with rules."""
        rules = [
            MCPPermissionRule(pattern="filesystem.*", allow=True),
            MCPPermissionRule(pattern="*.delete_*", allow=False),
        ]
        manager = MCPPermissionManager(rules=rules)

        assert len(manager.rules) == 2
        assert manager.rules[0].pattern == "filesystem.*"


# =============================================================================
# MCPPermissionManager.check_access Tests
# =============================================================================


class TestMCPPermissionManagerCheckAccess:
    """Tests for MCPPermissionManager.check_access method."""

    def test_deny_by_default(self):
        """Test access denied by default when deny_by_default=True."""
        manager = MCPPermissionManager(deny_by_default=True)

        result = manager.check_access("unknown_server", "unknown_tool")

        assert result is False

    def test_allow_by_default(self):
        """Test access allowed by default when deny_by_default=False."""
        manager = MCPPermissionManager(deny_by_default=False)

        result = manager.check_access("unknown_server", "unknown_tool")

        assert result is True

    def test_exact_pattern_match(self):
        """Test exact pattern matching."""
        manager = MCPPermissionManager()
        manager.add_rule("filesystem.read_file", allow=True)

        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("filesystem", "write_file") is False

    def test_server_wildcard(self):
        """Test server.* pattern matches all tools on server."""
        manager = MCPPermissionManager()
        manager.add_rule("filesystem.*", allow=True)

        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("filesystem", "write_file") is True
        assert manager.check_access("github", "read_file") is False

    def test_tool_wildcard(self):
        """Test *.tool pattern matches tool on any server."""
        manager = MCPPermissionManager()
        manager.add_rule("*.read_file", allow=True)

        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("github", "read_file") is True
        assert manager.check_access("filesystem", "write_file") is False

    def test_full_wildcard(self):
        """Test *.* pattern matches everything."""
        manager = MCPPermissionManager()
        manager.add_rule("*.*", allow=True)

        assert manager.check_access("any_server", "any_tool") is True
        assert manager.check_access("other", "other_tool") is True

    def test_glob_prefix_pattern(self):
        """Test glob pattern with prefix matching."""
        manager = MCPPermissionManager()
        manager.add_rule("filesystem.read_*", allow=True)

        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("filesystem", "read_dir") is True
        assert manager.check_access("filesystem", "write_file") is False

    def test_deny_rule_overrides_default_allow(self):
        """Test deny rule takes precedence when deny_by_default=False."""
        manager = MCPPermissionManager(deny_by_default=False)
        manager.add_rule("*.dangerous_*", allow=False)

        assert manager.check_access("any", "dangerous_delete") is False
        assert manager.check_access("any", "safe_tool") is True

    def test_first_matching_rule_wins(self):
        """Test that first matching rule wins."""
        manager = MCPPermissionManager()
        manager.add_rule("filesystem.read_file", allow=True)
        manager.add_rule("filesystem.*", allow=False)

        # First rule matches, so allowed
        assert manager.check_access("filesystem", "read_file") is True
        # Second rule matches write_file
        assert manager.check_access("filesystem", "write_file") is False

    def test_deny_rule_before_allow_rule(self):
        """Test deny rule placed before allow rule blocks access."""
        manager = MCPPermissionManager()
        manager.add_rule("filesystem.secret_*", allow=False)
        manager.add_rule("filesystem.*", allow=True)

        assert manager.check_access("filesystem", "secret_key") is False
        assert manager.check_access("filesystem", "read_file") is True


# =============================================================================
# MCPPermissionManager Temporary Grants Tests
# =============================================================================


class TestMCPPermissionManagerTemporaryGrants:
    """Tests for temporary grant functionality."""

    def test_grant_specific_tool(self):
        """Test granting access to a specific tool."""
        manager = MCPPermissionManager()

        manager.grant_temporary("filesystem", "read_file")

        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("filesystem", "write_file") is False

    def test_grant_all_server_tools(self):
        """Test granting access to all tools on a server."""
        manager = MCPPermissionManager()

        manager.grant_temporary("filesystem")

        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("filesystem", "write_file") is True
        assert manager.check_access("github", "read_file") is False

    def test_grant_all_tools(self):
        """Test granting access to all tools via *.*."""
        manager = MCPPermissionManager()
        manager._temporary_grants.add("*.*")

        assert manager.check_access("any", "tool") is True
        assert manager.check_access("other", "other_tool") is True

    def test_temporary_grant_overrides_rules(self):
        """Test temporary grants override deny rules."""
        manager = MCPPermissionManager()
        manager.add_rule("filesystem.*", allow=False)

        # Before grant, denied
        assert manager.check_access("filesystem", "read_file") is False

        # After grant, allowed
        manager.grant_temporary("filesystem", "read_file")
        assert manager.check_access("filesystem", "read_file") is True

    def test_revoke_specific_tool(self):
        """Test revoking access to a specific tool."""
        manager = MCPPermissionManager()
        manager.grant_temporary("filesystem", "read_file")

        manager.revoke_temporary("filesystem", "read_file")

        assert manager.check_access("filesystem", "read_file") is False

    def test_revoke_all_server_tools(self):
        """Test revoking access to all server tools."""
        manager = MCPPermissionManager()
        manager.grant_temporary("filesystem")

        manager.revoke_temporary("filesystem")

        assert manager.check_access("filesystem", "read_file") is False

    def test_clear_temporary_grants(self):
        """Test clearing all temporary grants."""
        manager = MCPPermissionManager()
        manager.grant_temporary("filesystem", "read_file")
        manager.grant_temporary("github", "push_code")
        manager.grant_temporary("slack")

        manager.clear_temporary_grants()

        assert manager.check_access("filesystem", "read_file") is False
        assert manager.check_access("github", "push_code") is False
        assert manager.check_access("slack", "send_message") is False


# =============================================================================
# MCPPermissionManager Rule Management Tests
# =============================================================================


class TestMCPPermissionManagerRuleManagement:
    """Tests for rule management methods."""

    def test_add_rule(self):
        """Test add_rule adds to rules list."""
        manager = MCPPermissionManager()

        manager.add_rule("filesystem.*", allow=True)

        assert len(manager.rules) == 1
        assert manager.rules[0].pattern == "filesystem.*"
        assert manager.rules[0].allow is True

    def test_add_multiple_rules(self):
        """Test adding multiple rules preserves order."""
        manager = MCPPermissionManager()

        manager.add_rule("filesystem.*", allow=True)
        manager.add_rule("github.*", allow=True)
        manager.add_rule("*.delete_*", allow=False)

        assert len(manager.rules) == 3
        assert manager.rules[0].pattern == "filesystem.*"
        assert manager.rules[1].pattern == "github.*"
        assert manager.rules[2].pattern == "*.delete_*"

    def test_allow_server_shortcut(self):
        """Test allow_server convenience method."""
        manager = MCPPermissionManager()

        manager.allow_server("filesystem")

        assert manager.check_access("filesystem", "any_tool") is True

    def test_deny_server_shortcut(self):
        """Test deny_server convenience method."""
        manager = MCPPermissionManager(deny_by_default=False)

        manager.deny_server("dangerous_server")

        assert manager.check_access("dangerous_server", "any_tool") is False
        assert manager.check_access("safe_server", "any_tool") is True


# =============================================================================
# MCPPermissionManager Edge Cases
# =============================================================================


class TestMCPPermissionManagerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_server_name(self):
        """Test with empty server name."""
        manager = MCPPermissionManager()
        manager.add_rule("*.*", allow=True)

        # Should still match
        assert manager.check_access("", "tool") is True

    def test_empty_tool_name(self):
        """Test with empty tool name."""
        manager = MCPPermissionManager()
        manager.add_rule("*.*", allow=True)

        # Should still match
        assert manager.check_access("server", "") is True

    def test_special_characters_in_names(self):
        """Test with special characters in server/tool names."""
        manager = MCPPermissionManager()
        manager.add_rule("my-server_v2.read-file_v1", allow=True)

        assert manager.check_access("my-server_v2", "read-file_v1") is True

    def test_multiple_temporary_grants_for_same_tool(self):
        """Test granting the same tool multiple times."""
        manager = MCPPermissionManager()

        manager.grant_temporary("filesystem", "read_file")
        manager.grant_temporary("filesystem", "read_file")  # Duplicate

        # Should still work
        assert manager.check_access("filesystem", "read_file") is True

        # One revoke should be enough
        manager.revoke_temporary("filesystem", "read_file")
        assert manager.check_access("filesystem", "read_file") is False

    def test_revoke_nonexistent_grant(self):
        """Test revoking a grant that doesn't exist."""
        manager = MCPPermissionManager()

        # Should not raise
        manager.revoke_temporary("nonexistent", "tool")

    def test_question_mark_wildcard(self):
        """Test fnmatch ? wildcard for single character matching."""
        manager = MCPPermissionManager()
        manager.add_rule("filesystem.read_?ile", allow=True)

        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("filesystem", "read_bile") is True
        assert manager.check_access("filesystem", "read_files") is False  # Too long

    def test_bracket_pattern(self):
        """Test fnmatch bracket pattern for character sets."""
        manager = MCPPermissionManager()
        manager.add_rule("filesystem.[rw]*", allow=True)

        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("filesystem", "write_file") is True
        assert manager.check_access("filesystem", "delete_file") is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestMCPPermissionManagerIntegration:
    """Integration tests for realistic permission scenarios."""

    def test_realistic_setup(self):
        """Test a realistic permission setup for an IDE."""
        manager = MCPPermissionManager(deny_by_default=True)

        # Allow filesystem operations except dangerous ones
        manager.add_rule("filesystem.delete_*", allow=False)
        manager.add_rule("filesystem.*", allow=True)

        # Allow read-only GitHub operations
        manager.add_rule("github.list_*", allow=True)
        manager.add_rule("github.get_*", allow=True)
        manager.add_rule("github.search_*", allow=True)

        # Verify filesystem access
        assert manager.check_access("filesystem", "read_file") is True
        assert manager.check_access("filesystem", "write_file") is True
        assert manager.check_access("filesystem", "delete_file") is False
        assert manager.check_access("filesystem", "delete_directory") is False

        # Verify GitHub access
        assert manager.check_access("github", "list_repos") is True
        assert manager.check_access("github", "get_repo") is True
        assert manager.check_access("github", "search_code") is True
        assert manager.check_access("github", "push_code") is False

        # Verify unknown servers blocked
        assert manager.check_access("unknown", "any_tool") is False

    def test_one_time_allow_workflow(self):
        """Test the 'allow once' workflow for user prompts."""
        manager = MCPPermissionManager(deny_by_default=True)

        # User is prompted for github.push_code
        assert manager.check_access("github", "push_code") is False

        # User approves for this session
        manager.grant_temporary("github", "push_code")
        assert manager.check_access("github", "push_code") is True

        # Other GitHub tools still blocked
        assert manager.check_access("github", "delete_repo") is False

        # At end of session, clear grants
        manager.clear_temporary_grants()
        assert manager.check_access("github", "push_code") is False

    def test_server_whitelisting_workflow(self):
        """Test whitelisting an entire server."""
        manager = MCPPermissionManager(deny_by_default=True)

        # User trusts a server completely
        manager.allow_server("trusted_server")

        assert manager.check_access("trusted_server", "any_tool") is True
        assert manager.check_access("trusted_server", "another_tool") is True
        assert manager.check_access("untrusted_server", "any_tool") is False
