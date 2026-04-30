"""List all local and remote git branches for the repository."""

import subprocess
import sys


def list_branches() -> dict:
    """Return all local and remote git branches.

    Returns:
        dict: A dictionary with keys ``"local"`` and ``"remote"``, each
            containing a list of branch name strings.

    Raises:
        RuntimeError: If git is not available or the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "branch", "-a"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git branch -a failed: {exc.stderr.strip()}") from exc

    local_branches = []
    remote_branches = []

    for line in result.stdout.splitlines():
        branch = line.strip().lstrip("* ").strip()
        if not branch:
            continue
        if branch.startswith("remotes/"):
            # Strip the leading "remotes/" prefix for readability
            remote_branches.append(branch.removeprefix("remotes/"))
        else:
            local_branches.append(branch)

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
