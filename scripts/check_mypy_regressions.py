from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ERROR_PATTERN = re.compile(r"^(.*?):\d+: error: (.*?)\s+\[([^\]]+)\]$")


def run_mypy(cwd: Path) -> tuple[int, Counter[tuple[str, str, str]]]:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental", "ai_platform"],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    errors: Counter[tuple[str, str, str]] = Counter()
    diagnostic_lines = [line for line in result.stdout.splitlines() if ": error:" in line]

    for line in diagnostic_lines:
        match = ERROR_PATTERN.match(line)
        if not match:
            print(result.stdout)
            raise RuntimeError(f"Could not parse a MyPy diagnostic in {cwd}: {line}")

        filename, message, code = match.groups()
        errors[(filename, message, code)] += 1

    if len(diagnostic_lines) != sum(errors.values()):
        raise RuntimeError(f"MyPy diagnostic parsing was incomplete in {cwd}.")

    # MyPy uses exit code 1 for type errors. Higher exit codes indicate
    # invocation/configuration failures that should not be baseline-accepted.
    if result.returncode not in (0, 1):
        print(result.stdout)
        raise RuntimeError(
            f"MyPy could not complete successfully in {cwd} " f"(exit code {result.returncode})."
        )

    if result.returncode == 1 and not errors:
        print(result.stdout)
        raise RuntimeError(f"MyPy failed in {cwd}, but no parseable diagnostics were found.")

    return result.returncode, errors


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/check_mypy_regressions.py BASE_COMMIT",
            file=sys.stderr,
        )
        return 2

    base_ref = sys.argv[1]
    root = ROOT

    verified_base = subprocess.run(
        ["git", "rev-parse", "--verify", f"{base_ref}^{{commit}}"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if verified_base.returncode != 0:
        print(
            f"Cannot resolve base commit {base_ref!r}:\n" f"{verified_base.stderr}",
            file=sys.stderr,
        )
        return 2

    base_sha = verified_base.stdout.strip()

    with tempfile.TemporaryDirectory(prefix="mypy-base-") as temp_dir:
        base_worktree = Path(temp_dir) / "base"

        add = subprocess.run(
            ["git", "worktree", "add", "--detach", str(base_worktree), base_sha],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if add.returncode != 0:
            print(add.stdout)
            print(add.stderr, file=sys.stderr)
            return 2

        try:
            print(f"Checking base commit: {base_sha}")
            _, base_errors = run_mypy(base_worktree)

            print(f"Checking PR commit: {subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=root, text=True
            ).strip()}")
            _, pr_errors = run_mypy(root)

            new_errors = pr_errors - base_errors
            resolved_errors = base_errors - pr_errors

            print()
            print(f"Base diagnostic count: {sum(base_errors.values())}")
            print(f"PR diagnostic count:   {sum(pr_errors.values())}")
            print(f"New diagnostic count:  {sum(new_errors.values())}")
            print(f"Resolved diagnostics:  {sum(resolved_errors.values())}")

            for (filename, message, code), count in sorted(new_errors.items()):
                print(f"NEW: {filename}: {message} [{code}] x{count}")

            if new_errors:
                print("\nMyPy regression check FAILED: the PR introduces " "new diagnostics.")
                return 1

            print(
                "\nMyPy regression check PASSED: no new diagnostics "
                "compared with the base commit."
            )
            return 0
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(base_worktree)],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )


if __name__ == "__main__":
    raise SystemExit(main())
