"""Checks AgentGraph's stepping, gating and persistence without calling the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from rooster import AgentGraph, GraphError, Run


class FakeAgent:
    """Stands in for Agent, recording what it was asked and inventing a session id."""

    def __init__(self, name: str, cwd: Path, session_id: str | None = None) -> None:
        self.name = name
        self.cwd = cwd
        self.session_id = session_id
        self.prompts: list[str] = []
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    def run(self, prompt: str) -> dict:
        """Pretends to run a turn, so the graph has something to collect usage from."""
        self.prompts.append(prompt)
        self.session_id = self.session_id or f"session-for-{self.name}"
        self.total_cost_usd += 0.01
        self.total_tokens += 100
        return {"ok": True, "text": f"{self.name} says ok", "session_id": self.session_id}


@pytest.fixture
def graph(tmp_path: Path) -> AgentGraph:
    """Builds a two-step graph with a gate between the steps."""
    return build_graph(tmp_path)


def build_graph(tmp_path: Path) -> AgentGraph:
    """Returns a two-step graph with a gate between the steps, and fresh agents."""
    agents = {"writer": FakeAgent("writer", tmp_path), "tester": FakeAgent("tester", tmp_path)}
    g = AgentGraph("bdd", agents, start="write", state_dir=tmp_path / ".rooster")

    @g.step("write", gate=True)
    def write(run: Run) -> str:
        run.data["features"] = agents["writer"].run(run.prompt)["text"]
        return "test"

    @g.step("test")
    def test(run: Run) -> None:
        run.data["tests"] = agents["tester"].run("write tests")["text"]
        return None

    return g


def test_a_run_stops_at_the_first_gate(graph: AgentGraph) -> None:
    """Proves the graph parks at a gate instead of running to the end."""
    run = graph.start("probe view shows a channel heat map")

    assert run.waiting
    assert run.cursor == "test"
    assert "features" in run.data
    # the step after the gate must not have run
    assert "tests" not in run.data


def test_resume_continues_from_the_cursor(graph: AgentGraph) -> None:
    """Proves a parked run picks up where it stopped and then finishes."""
    started = graph.start("a feature")

    resumed = graph.resume(started.slug)

    assert resumed.status == "done"
    assert resumed.cursor is None
    assert resumed.data["tests"] == "tester says ok"


def test_session_ids_are_saved_and_handed_back(graph: AgentGraph, tmp_path: Path) -> None:
    """Proves the graph carries agent sessions across a stop, which is the point of the gate."""
    started = graph.start("a feature")
    assert started.sessions["writer"] == "session-for-writer"

    # simulate a new process: the agents are fresh objects with no memory of a session
    for agent in graph.agents.values():
        agent.session_id = None

    graph.resume(started.slug)

    assert graph.agents["writer"].session_id == "session-for-writer"


def test_sessions_are_tagged_with_their_run(
    graph: AgentGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves each agent's session is tagged with the graph and run it belongs to."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        "rooster.graph.tag_session",
        lambda session_id, tag, directory=None: calls.append((session_id, tag, directory)),
    )

    started = graph.start("a feature")

    tag = f"rooster:bdd/{started.slug}"
    assert calls == [("session-for-writer", tag, str(tmp_path))]

    # the next step tags the writer again and the tester for the first time
    calls.clear()
    finished = graph.resume(started.slug)

    assert finished.status == "done"
    assert ("session-for-writer", tag, str(tmp_path)) in calls
    assert ("session-for-tester", tag, str(tmp_path)) in calls


def test_a_session_that_cannot_be_tagged_does_not_fail_the_run(
    graph: AgentGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves a missing session file leaves the run to gate and finish as usual."""

    def missing(session_id: str, tag: str, directory: str | None = None) -> None:
        raise FileNotFoundError(session_id)

    monkeypatch.setattr("rooster.graph.tag_session", missing)

    started = graph.start("a feature")
    assert started.waiting

    finished = graph.resume(started.slug)
    assert finished.status == "done"
    assert finished.failure is None


def test_state_survives_a_new_graph_object(graph: AgentGraph, tmp_path: Path) -> None:
    """Proves the run is genuinely on disk, not merely in memory."""
    started = graph.start("a feature")

    reread = AgentGraph(
        "bdd", {}, start="write", state_dir=tmp_path / ".rooster"
    ).status(started.slug)

    assert reread.slug == started.slug
    assert reread.cursor == "test"
    assert reread.status == "awaiting_human"


def test_usage_is_totalled_across_agents(graph: AgentGraph) -> None:
    """Proves the run accumulates what every agent spent, not just the last one."""
    started = graph.start("a feature")
    finished = graph.resume(started.slug)

    assert finished.tokens == 200
    assert finished.cost_usd == pytest.approx(0.02)


def test_usage_accumulates_across_a_new_process(tmp_path: Path) -> None:
    """Proves a resume in a new process adds to the saved totals instead of replacing them."""
    started = build_graph(tmp_path).start("a feature")

    # a new process builds new agents, whose totals start at zero
    finished = build_graph(tmp_path).resume(started.slug)

    assert finished.tokens == 200
    assert finished.cost_usd == pytest.approx(0.02)


def test_runs_sharing_agents_do_not_count_each_others_spend(graph: AgentGraph) -> None:
    """Proves each run counts only what was spent on it, though the agents are shared."""
    first = graph.start("a feature")
    second = graph.start("another feature")

    assert first.tokens == 100
    assert second.tokens == 100


def test_a_halted_step_still_counts_its_spend(tmp_path: Path) -> None:
    """Proves a step that spends and then raises leaves its spend on the saved run."""
    agents = {"writer": FakeAgent("writer", tmp_path)}
    g = AgentGraph("x", agents, start="boom", state_dir=tmp_path / ".rooster")

    @g.step("boom")
    def boom(run: Run) -> None:
        agents["writer"].run("anything")
        raise ValueError("the agent wrote nonsense")

    with pytest.raises(ValueError):
        g.start("a feature")

    saved = g.runs()[0]
    assert saved.status == "halted"
    assert saved.cost_usd == pytest.approx(0.01)


def test_slugs_do_not_collide(graph: AgentGraph) -> None:
    """Proves two runs started from the same words get separate files."""
    first = graph.start("the same feature")
    second = graph.start("the same feature")

    assert first.slug == "the-same-feature"
    assert second.slug == "the-same-feature-2"
    assert len(graph.runs()) == 2


def test_a_step_that_raises_halts_and_is_saved(tmp_path: Path) -> None:
    """Proves a crash inside a step is recorded on disk before the exception escapes."""
    g = AgentGraph("x", {}, start="boom", state_dir=tmp_path / ".rooster")

    @g.step("boom")
    def boom(run: Run) -> None:
        raise ValueError("the agent wrote nonsense")

    with pytest.raises(ValueError):
        g.start("a feature")

    saved = g.runs()[0]
    assert saved.status == "halted"
    assert "ValueError" in saved.failure


def test_an_unknown_next_step_halts(tmp_path: Path) -> None:
    """Proves a typo in a returned step name stops the run rather than failing silently."""
    g = AgentGraph("x", {}, start="one", state_dir=tmp_path / ".rooster")

    @g.step("one")
    def one(run: Run) -> str:
        return "tow"  # typo for "two"

    run = g.start("a feature")

    assert run.status == "halted"
    assert "tow" in run.failure


def test_a_runaway_loop_is_capped(tmp_path: Path) -> None:
    """Proves a cycle stops loudly instead of running forever."""
    g = AgentGraph("x", {}, start="again", state_dir=tmp_path / ".rooster", max_steps=5)

    @g.step("again")
    def again(run: Run) -> str:
        run.revisions += 1
        return "again"

    run = g.start("a feature")

    assert run.status == "halted"
    assert "5 steps" in run.failure
    assert run.revisions == 5


def test_an_unregistered_start_is_caught_before_any_agent_runs(tmp_path: Path) -> None:
    """Proves a misdeclared graph fails at once, not halfway through a run."""
    g = AgentGraph("x", {}, start="missing", state_dir=tmp_path / ".rooster")

    @g.step("present")
    def present(run: Run) -> None:
        return None

    with pytest.raises(GraphError, match="start step 'missing' is not registered"):
        g.start("a feature")


def test_a_duplicate_step_name_is_rejected(tmp_path: Path) -> None:
    """Proves two steps cannot quietly claim the same name."""
    g = AgentGraph("x", {}, start="one", state_dir=tmp_path / ".rooster")

    @g.step("one")
    def one(run: Run) -> None:
        return None

    with pytest.raises(GraphError, match="already registered"):

        @g.step("one")
        def one_again(run: Run) -> None:
            return None


def test_resuming_a_finished_run_is_refused(graph: AgentGraph) -> None:
    """Proves a completed run cannot be restarted by accident."""
    started = graph.start("a feature")
    graph.resume(started.slug)

    with pytest.raises(GraphError, match="already done"):
        graph.resume(started.slug)


def test_an_unknown_run_names_the_ones_that_exist(graph: AgentGraph) -> None:
    """Proves a mistyped slug gets a useful error rather than a stack trace."""
    graph.start("a real feature")

    with pytest.raises(GraphError, match="no run named 'nope'"):
        graph.resume("nope")
