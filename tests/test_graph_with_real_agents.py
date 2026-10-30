"""Drives two real agents around a loop, proving AgentGraph works against the actual CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from rooster import Agent, AgentGraph, Job, JobStatus, Tool

AGENTS = Path(__file__).parent / "agents"
ROUNDS = 2

pytestmark = pytest.mark.live


def test_hello_and_goodbye_alternate_until_the_round_count_is_met(tmp_path: Path) -> None:
    """Proves the graph hands control back and forth between two agents until a count is met."""
    logs = tmp_path / "logs"
    hello = Agent("hello", AGENTS / "hello.md", cwd=tmp_path, log_dir=logs, remembers=True,
                  tools=Tool.NONE)
    goodbye = Agent("goodbye", AGENTS / "goodbye.md", cwd=tmp_path, log_dir=logs, remembers=True,
                    tools=Tool.NONE)

    graph = AgentGraph(
        "greeting",
        {"hello": hello, "goodbye": goodbye},
        start="say-hello",
        state_dir=tmp_path / ".rooster",
    )

    @graph.step("say-hello", next=("say-goodbye",))
    def say_hello(job: Job) -> str:
        """Asks the hello agent to speak, then hands over to goodbye."""
        result = hello.run("Go.")
        assert result["ok"], result["failure"]
        job.data.setdefault("transcript", []).append(result["text"].strip())
        return "say-goodbye"

    @graph.step("say-goodbye", next=("say-hello", None))
    def say_goodbye(job: Job) -> str | None:
        """Asks the goodbye agent to speak, then loops back until the rounds are done."""
        result = goodbye.run("Go.")
        assert result["ok"], result["failure"]
        job.data["transcript"].append(result["text"].strip())
        job.data["rounds"] = job.data.get("rounds", 0) + 1
        return "say-hello" if job.data["rounds"] < ROUNDS else None

    job = graph.start("greet two times")

    print("\n" + "\n".join(f"  {i + 1:2}. {line}" for i, line in enumerate(job.data["transcript"])))
    print(f"\n  rounds: {job.data['rounds']}   tokens: {job.tokens:,}   cost: ${job.cost_usd:.4f}")
    print("\n" + graph.diagram())

    assert job.status == JobStatus.DONE
    assert job.next_step is None
    assert job.data["rounds"] == ROUNDS
    assert job.data["transcript"] == ["Hello!", "Goodbye!"] * ROUNDS

    # each agent wrote one log per turn, and kept one conversation across all of them
    assert len(list(logs.glob("*-hello.log"))) == ROUNDS
    assert len(list(logs.glob("*-goodbye.log"))) == ROUNDS
    assert job.sessions["hello"] == hello.session_id
    assert job.sessions["goodbye"] == goodbye.session_id
    assert hello.session_id != goodbye.session_id

    # the job is on disk, finished, with the loop recorded
    saved = graph.load(job.slug)
    assert saved.status == JobStatus.DONE
    assert saved.data["transcript"] == job.data["transcript"]
