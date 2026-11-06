"""Runs the README's workflow on a tiny project: a writer changes the code, the tests check it,
and a reviewer judges it. It pauses for you before it starts and after each review.

Run it from the repository's root with `uv run python examples/review/run.py`. It copies
project/ into work/ first, so every run starts from the same code, and you can look at what the
writer changed in work/ afterwards. It uses a little of your Claude plan.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from rooster import Agent, AgentGraph, Command, Job, Tool

# the folder this example lives in
HERE = Path(__file__).resolve().parent

# the task the writer is given
TASK = "Add a subtract(a, b) function to calc.py, with a test."


class Verdict(BaseModel):
    """Holds the reviewer's answer: whether the change is accepted, and why."""

    accepted: bool
    reason: str


def build(work: Path) -> AgentGraph:
    """Returns the workflow, with a writer and a reviewer that both work in 'work'."""
    writer = Agent(
        "writer",
        HERE / "profiles" / "writer.md",
        cwd=work,
        log_dir=HERE / ".rooster" / "logs",
        remembers=True,
        tools=Tool.AUTHOR,
        commands=Command.PYTEST,
        editable=("*.py",),
        confined_to=work,
        max_work_usd=1.0,
    )
    reviewer = Agent(
        "reviewer",
        HERE / "profiles" / "reviewer.md",
        cwd=work,
        log_dir=HERE / ".rooster" / "logs",
        tools=Tool.INSPECT,
        confined_to=work,
        schema=Verdict,
        max_work_usd=0.5,
    )
    graph = AgentGraph("example", {"writer": writer, "reviewer": reviewer},
                       start="write", state_dir=HERE / ".rooster")

    @graph.step("write", next=("test",))
    def write(job: Job) -> str:
        feedback = job.data.pop("feedback", "")
        writer.run(f"{job.prompt}\n\n{feedback}".strip())
        return "test"

    @graph.step("test", next=("review", "write"))
    def test(job: Job) -> str:
        tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=work,
                               capture_output=True, text=True)
        if tests.returncode == 0:
            return "review"
        job.data["feedback"] = f"The tests failed:\n{tests.stdout[-2000:]}"
        return "write"

    @graph.step("review", next=("write", None), gate=True)
    def review(job: Job) -> str | None:
        result = reviewer.run(f"The task was: {job.prompt}\nReview the change.")
        if not result["ok"]:
            raise RuntimeError(result["failure"])
        verdict = result["data"]
        print(f"reviewer: {'accepted' if verdict.accepted else 'rejected'}. {verdict.reason}")
        if verdict.accepted:
            return None
        job.data["feedback"] = f"The reviewer rejected the change: {verdict.reason}"
        return "write"

    return graph


def main() -> None:
    # copy the starting project into work/, so each run starts from the same code
    work = HERE / "work"
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(HERE / "project", work)
    input(f"The project is copied to {work}. Open calc.py there: it has only add(). "
          "Press Enter to start.")

    graph = build(work)
    job = graph.start(TASK)
    while job.waiting:
        input("Paused after the review. Look at work/ if you like, then press Enter to go on.")
        job = graph.resume(job.slug)
    print(f"The job is {job.status}. It used ${job.cost_usd:.2f} at API prices.")
    if job.halt_reason:
        print(job.halt_reason)


if __name__ == "__main__":
    main()
