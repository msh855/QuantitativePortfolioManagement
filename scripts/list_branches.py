"""List all local and remote git branches for the repository."""

import subprocess
import sys


def list_branches() -> dict:
    """Return all local and remote git branches.

    Uses ``git for-each-ref`` (plumbing) instead of ``git branch -a``
    (porcelain) so branch names never need to be recovered from free-form,
    human-oriented text. That avoids a whole class of parsing pitfalls:
    the current-branch/worktree markers ("* "/"+ "), the detached-HEAD
    status row (e.g. "* (HEAD detached at <sha>)"), and the symbolic
    "origin/HEAD -> origin/main" ref, none of which are real branches.

    Returns:
        dict: A dictionary with keys ``"local"`` and ``"remote"``, each
            containing a list of branch name strings.

    Raises:
        RuntimeError: If git is not available or the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname)\t%(symref)", "refs/heads/", "refs/remotes/"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git for-each-ref failed: {exc.stderr.strip()}") from exc

    local_branches = []
    remote_branches = []

    for line in result.stdout.splitlines():
        if not line:
            continue
        refname, _, symref = line.partition("\t")
        if symref:
            # Skip symbolic refs, e.g. refs/remotes/origin/HEAD -> origin/main
            continue
        if refname.startswith("refs/heads/"):
            local_branches.append(refname.removeprefix("refs/heads/"))
        elif refname.startswith("refs/remotes/"):
            remote_branches.append(refname.removeprefix("refs/remotes/"))

    return {"local": local_branches, "remote": remote_branches}


def _print_branches(branches: dict) -> None:
    print("Local branches:")
    for b in branches["local"]:
        print(f"  {b}")

    print("\nRemote branches:")
    for b in branches["remote"]:
        print(f"  {b}")


if __name__ == "__main__":
    try:
        branches = list_branches()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_branches(branches)
