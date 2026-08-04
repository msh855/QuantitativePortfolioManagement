"""
Unit tests for scripts.list_branches module.

Tests cover:
- list_branches returns the expected dict structure
- local and remote branch lists are non-empty
- list_branches correctly parses git's "* "/"+ " markers and skips
  symbolic refs (e.g. "remotes/origin/HEAD -> origin/main")
- list_branches raises RuntimeError when git is unavailable or fails
- _print_branches outputs the expected sections
"""

import subprocess
import sys
import os
import pytest

# Allow importing the scripts directory directly without installing it as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from list_branches import list_branches, _print_branches  # noqa: E402


def _fake_run(stdout):
    """Build a stand-in for subprocess.run that returns fixed branch -a output."""

    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=0, stdout=stdout, stderr="")

    return _run


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

    def test_skips_symbolic_head_ref(self, monkeypatch):
        """The 'remotes/origin/HEAD -> origin/main' symbolic ref must be skipped."""
        stdout = "* main\n  feature/x\n  remotes/origin/main\n  remotes/origin/HEAD -> origin/main\n"
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout))

        result = list_branches()

        assert result == {"local": ["main", "feature/x"], "remote": ["origin/main"]}

    def test_strips_worktree_marker(self, monkeypatch):
        """Branches checked out in another worktree (prefixed with '+ ') are parsed correctly."""
        stdout = "  main\n+ feature/in-other-worktree\n"
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout))

        result = list_branches()

        assert result == {"local": ["main", "feature/in-other-worktree"], "remote": []}

    def test_raises_runtime_error_when_git_missing(self, monkeypatch):
        """A missing git executable must surface as RuntimeError."""

        def _raise_not_found(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", _raise_not_found)

        with pytest.raises(RuntimeError, match="git executable not found"):
            list_branches()

    def test_raises_runtime_error_when_git_fails(self, monkeypatch):
        """A non-zero exit from git branch -a must surface as RuntimeError."""

        def _raise_called_process_error(*args, **kwargs):
            raise subprocess.CalledProcessError(1, ["git", "branch", "-a"], stderr="not a git repository")

        monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

        with pytest.raises(RuntimeError, match="git branch -a failed"):
            list_branches()


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
