"""
Unit tests for scripts.list_branches module.

Tests cover:
- list_branches returns the expected dict structure
- list_branches correctly splits refs/heads/ vs refs/remotes/ entries
- list_branches skips symbolic refs (e.g. refs/remotes/origin/HEAD)
- list_branches raises RuntimeError when git is unavailable or fails
- _print_branches outputs the expected sections

All list_branches() tests stub subprocess.run with fixed, deterministic
`git for-each-ref` output so they don't depend on the branches/remotes
present in whatever checkout the suite happens to run in.
"""

import os
import subprocess
import sys

import pytest

# Allow importing the package without installing it: add the repository root
# (not scripts/ itself) so the documented `scripts.list_branches` import path
# is what actually gets exercised.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.list_branches import list_branches, _print_branches  # noqa: E402

SAMPLE_FOR_EACH_REF_STDOUT = (
    "refs/heads/main\t\n"
    "refs/heads/feature/x\t\n"
    "refs/remotes/origin/main\t\n"
    "refs/remotes/origin/HEAD\trefs/remotes/origin/main\n"
)


def _fake_run(stdout):
    """Build a stand-in for subprocess.run that returns fixed for-each-ref output."""

    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=0, stdout=stdout, stderr="")

    return _run


@pytest.fixture
def sample_branches(monkeypatch):
    """Patch subprocess.run with deterministic for-each-ref output and return the parsed result."""
    monkeypatch.setattr(subprocess, "run", _fake_run(SAMPLE_FOR_EACH_REF_STDOUT))
    return list_branches()


@pytest.mark.unit
class TestListBranches:
    """Tests for list_branches function."""

    def test_returns_dict_with_required_keys(self, sample_branches):
        """list_branches must return a dict with 'local' and 'remote' keys."""
        assert isinstance(sample_branches, dict)
        assert "local" in sample_branches
        assert "remote" in sample_branches

    def test_local_branches_is_list(self, sample_branches):
        """The 'local' value must be a list."""
        assert isinstance(sample_branches["local"], list)

    def test_remote_branches_is_list(self, sample_branches):
        """The 'remote' value must be a list."""
        assert isinstance(sample_branches["remote"], list)

    def test_branch_names_are_strings(self, sample_branches):
        """All branch names must be plain strings."""
        for branch in sample_branches["local"] + sample_branches["remote"]:
            assert isinstance(branch, str)
            assert branch  # non-empty string

    def test_local_and_remote_refs_split_correctly(self, sample_branches):
        """refs/heads/ entries land in 'local', refs/remotes/ entries land in 'remote'."""
        assert sample_branches["local"] == ["main", "feature/x"]
        assert sample_branches["remote"] == ["origin/main"]

    def test_no_refname_prefixes_leak_through(self, sample_branches):
        """Branch names must have their 'refs/heads/'/'refs/remotes/' prefix stripped."""
        for branch in sample_branches["local"] + sample_branches["remote"]:
            assert not branch.startswith("refs/")

    def test_skips_symbolic_head_ref(self, sample_branches):
        """The symbolic 'refs/remotes/origin/HEAD' ref must not appear as a branch."""
        assert "HEAD" not in sample_branches["remote"]
        assert len(sample_branches["remote"]) == 1

    def test_ignores_refs_outside_heads_and_remotes(self, monkeypatch):
        """Tags and other ref types must not be reported as branches."""
        stdout = "refs/heads/main\t\nrefs/tags/v1.0.0\t\n"
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout))

        result = list_branches()

        assert result == {"local": ["main"], "remote": []}

    def test_raises_runtime_error_when_git_missing(self, monkeypatch):
        """A missing git executable must surface as RuntimeError."""

        def _raise_not_found(*args, **kwargs):
            raise FileNotFoundError("git not found")

        monkeypatch.setattr(subprocess, "run", _raise_not_found)

        with pytest.raises(RuntimeError, match="git executable not found"):
            list_branches()

    def test_raises_runtime_error_when_git_fails(self, monkeypatch):
        """A non-zero exit from git for-each-ref must surface as RuntimeError."""

        def _raise_called_process_error(*args, **kwargs):
            raise subprocess.CalledProcessError(1, ["git", "for-each-ref"], stderr="not a git repository")

        monkeypatch.setattr(subprocess, "run", _raise_called_process_error)

        with pytest.raises(RuntimeError, match="git for-each-ref failed"):
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
