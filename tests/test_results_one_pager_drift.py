"""Drift test: docs/RESULTS_ONE_PAGER.md must be the generator's exact output.

The one-pager is generated evidence (scripts/generate_results_one_pager.py);
any hand edit or source-file change that is not regenerated shows up here.

Comparison basis: the committed git blob (``git show HEAD:docs/RESULTS_ONE_PAGER.md``),
not the working-tree bytes. Reason: this repository is checked out with
autocrlf on Windows, so the working-tree copy carries CRLF line endings while
the generator emits LF; the blob is what is actually committed and is
byte-identical across platforms. If git is unavailable, the test falls back
to the working-tree file with newlines normalised to LF (a weaker check that
still catches content drift).

Byte-stability contract: the generator is deterministic (sorted iteration,
verbatim source strings, no timestamps, LF newlines) — see its module
docstring.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATOR = REPO / "scripts" / "generate_results_one_pager.py"
ONE_PAGER = "docs/RESULTS_ONE_PAGER.md"


def _committed_bytes() -> bytes | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO), "show", f"HEAD:{ONE_PAGER}"],
            capture_output=True, check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _worktree_bytes_normalised() -> bytes:
    path = REPO / ONE_PAGER
    return path.read_bytes().replace(b"\r\n", b"\n")


def _generated_bytes(tmp_path: Path) -> bytes:
    out = tmp_path / "RESULTS_ONE_PAGER.md"
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(out)],
        capture_output=True, check=True,
    )
    assert result.returncode == 0
    return out.read_bytes()


def test_one_pager_matches_generator(tmp_path):
    committed = _committed_bytes()
    if committed is None:
        # no git (e.g. source distribution): fall back to the normalised
        # working-tree file and say so in the failure message if it drifts.
        committed = _worktree_bytes_normalised()
        basis = "newline-normalised working-tree file (git unavailable)"
    else:
        basis = "committed git blob"
    generated = _generated_bytes(tmp_path)
    assert generated == committed, (
        "docs/RESULTS_ONE_PAGER.md drifted from the generator output "
        f"(compared against the {basis}). Re-run "
        "`python scripts/generate_results_one_pager.py` and commit the "
        "regenerated file; do not hand-edit it."
    )
