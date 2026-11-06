"""Checks that the review example builds a valid workflow, without calling Claude."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

# the example's folder
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "review"


def load_example():
    """Imports examples/review/run.py as a module."""
    spec = importlib.util.spec_from_file_location("review_example", EXAMPLE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_builds_a_valid_workflow(tmp_path: Path) -> None:
    """Proves the example's agents and steps are valid, and its diagram has all three steps."""
    diagram = load_example().build(tmp_path).diagram()

    for step in ("write", "test", "review (gate)"):
        assert step in diagram


def test_the_example_project_passes_its_own_tests(tmp_path: Path) -> None:
    """Proves the starting project's test passes, so a failing run is the writer's doing."""
    shutil.copytree(EXAMPLE / "project", tmp_path / "work")

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                           cwd=tmp_path / "work", capture_output=True, text=True)

    assert tests.returncode == 0, tests.stdout
