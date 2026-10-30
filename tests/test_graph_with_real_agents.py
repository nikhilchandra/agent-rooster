"""Drives two real agents around a loop, proving AgentGraph works against the actual CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from rooster import Agent, AgentGraph, Run, Tool

AGENTS = Path(__file__).parent / "agents"
ROUNDS = 2

pytestmark = pytest.mark.live


def test_hello_and_goodbye_alternate_until_the_round_count_is_met(tmp_path: Path) -> None:
    """Proves the graph hands control back and forth between two agents until a count is met."""
    logs = tmp_path / "logs"
    hello = Agent("hello", AGENTS / "hello.md", cwd=tmp_path, log_dir=logs, tools=Tool.NONE)
    goodbye = Agent("goodbye", AGENTS / "goodbye.md", cwd=tmp_path, log_dir=logs, tools=Tool.NONE)

    graph = AgentGraph(
        "greeting",
        {"hello": hello, "goodbye": goodbye},
        start="say-hello",
        state_dir=tmp_path / ".rooster",
    )

    @graph.step("say-hello")
    def say_hello(run: Run) -> str:
        """Asks the hello agent to speak, then hands over to goodbye."""
        result = hello.run("Go.")
        assert result["ok"], result["failure"]
        run.data.setdefault("transcript", []).append(result["text"].strip())
        return "say-goodbye"

    @graph.step("say-goodbye")
    def say_goodbye(run: Run) -> str | None:
        """Asks the goodbye agent to speak, then loops back until five rounds are done."""
        result = goodbye.run("Go.")
        assert result["ok"], result["failure"]
        run.data["transcript"].append(result["text"].strip())
        run.data["rounds"] = run.data.get("rounds", 0) + 1
        return "say-hello" if run.data["rounds"] < ROUNDS else None

    run = graph.start("greet two times")

    print("\n" + "\n".join(f"  {i + 1:2}. {line}" for i, line in enumerate(run.data["transcript"])))
    print(f"\n  rounds: {run.data['rounds']}   tokens: {run.tokens:,}   cost: ${run.cost_usd:.4f}")

    assert run.status == "done"
    assert run.cursor is None
    assert run.data["rounds"] == ROUNDS
    assert run.data["transcript"] == ["Hello!", "Goodbye!"] * ROUNDS

    # each agent wrote one log per turn, and kept one conversation across all of them
    assert len(list(logs.glob("*-hello.log"))) == ROUNDS
    assert len(list(logs.glob("*-goodbye.log"))) == ROUNDS
    assert run.sessions["hello"] == hello.session_id
    assert run.sessions["goodbye"] == goodbye.session_id
    assert hello.session_id != goodbye.session_id

    # the run is on disk, finished, with the loop recorded
    saved = graph.status(run.slug)
    assert saved.status == "done"
    assert saved.data["transcript"] == run.data["transcript"]
