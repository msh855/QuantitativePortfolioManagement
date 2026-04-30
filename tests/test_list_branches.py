"""
Unit tests for scripts.list_branches module.

Tests cover:
- list_branches returns the expected dict structure
- local and remote branch lists are non-empty
- _print_branches outputs the expected sections
"""

import sys
import os
import pytest

# Allow importing the scripts directory directly without installing it as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from list_branches import list_branches, _print_branches  # noqa: E402


@pytest.mark.unit
class TestListBranches:
    """Tests for list_branches function."""

    def test_returns_dict_with_required_keys(self):
        """list_branches must return a dict with 'local' and 'remote' keys."""
        result = list_branches()
        assert isinstance(result, dict)
        assert "local" in result
        assert "remote" in result

    def test_local_branches_is_list(self):
        """The 'local' value must be a list."""
        result = list_branches()
        assert isinstance(result["local"], list)

    def test_remote_branches_is_list(self):
        """The 'remote' value must be a list."""
        result = list_branches()
        assert isinstance(result["remote"], list)

    def test_local_branches_non_empty(self):
        """There must be at least one local branch."""
        result = list_branches()
        assert len(result["local"]) >= 1

    def test_branch_names_are_strings(self):
        """All branch names must be plain strings."""
        result = list_branches()
        for branch in result["local"] + result["remote"]:
            assert isinstance(branch, str)
            assert branch  # non-empty string

    def test_local_branches_have_no_remotes_prefix(self):
        """Local branches must not start with 'remotes/'."""
        result = list_branches()
        for branch in result["local"]:
            assert not branch.startswith("remotes/")

    def test_remote_branches_have_no_remotes_prefix(self):
        """The 'remotes/' prefix must have been stripped from remote entries."""
        result = list_branches()
        for branch in result["remote"]:
            assert not branch.startswith("remotes/")


@pytest.mark.unit
class TestPrintBranches:
    """Tests for _print_branches helper."""

    def test_prints_local_and_remote_sections(self, capsys):
        """_print_branches must print both section headers."""
        branches = {"local": ["main"], "remote": ["origin/main"]}
        _print_branches(branches)
        captured = capsys.readouterr()
        assert "Local branches:" in captured.out
        assert "Remote branches:" in captured.out
        assert "main" in captured.out
        assert "origin/main" in captured.out

    def test_handles_empty_branches(self, capsys):
        """_print_branches must not raise when branch lists are empty."""
        branches = {"local": [], "remote": []}
        _print_branches(branches)
        captured = capsys.readouterr()
        assert "Local branches:" in captured.out
        assert "Remote branches:" in captured.out
