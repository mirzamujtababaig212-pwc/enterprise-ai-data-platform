from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_rag_release_evaluation.py"


def test_rag_release_evaluation_cli_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Quality gate:        PASS" in result.stdout
    assert "Release decision:    PASS" in result.stdout
    assert "Dataset:             vehicle-retrieval" in result.stdout
    assert "Dataset version:     v2" in result.stdout
    assert "Evaluated queries:   7" in result.stdout
