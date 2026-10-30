"""Checks AgentGraph's stepping, gating and persistence without calling the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from rooster import AgentGraph, GraphError, Job, JobStatus


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

    @g.step("write", next=("test",), gate=True)
    def write(job: Job) -> str:
        job.data["features"] = agents["writer"].run(job.prompt)["text"]
        return "test"

    @g.step("test", next=(None,))
    def test(job: Job) -> None:
        job.data["tests"] = agents["tester"].run("write tests")["text"]
        return None

    return g


def test_a_job_stops_at_the_first_gate(graph: AgentGraph) -> None:
    """Proves the graph pauses at a gate instead of running to the end."""
    job = graph.start("probe view shows a channel heat map")

    assert job.waiting
    assert job.status == JobStatus.WAITING
    assert job.next_step == "test"
    assert "features" in job.data
    # the step after the gate must not have run
    assert "tests" not in job.data


def test_resume_continues_from_the_next_step(graph: AgentGraph) -> None:
    """Proves a paused job picks up where it stopped and then finishes."""
    started = graph.start("a feature")

    resumed = graph.resume(started.slug)

    assert resumed.status == JobStatus.DONE
    assert resumed.next_step is None
    assert resumed.data["tests"] == "tester says ok"


def test_session_ids_are_saved_and_handed_back(graph: AgentGraph) -> None:
    """Proves the graph carries agent sessions across a stop, which is the point of the gate."""
    started = graph.start("a feature")
    assert started.sessions["writer"] == "session-for-writer"

    # simulate a new process: the agents are fresh objects with no memory of a session
    for agent in graph.agents.values():
        agent.session_id = None

    graph.resume(started.slug)

    assert graph.agents["writer"].session_id == "session-for-writer"


def test_state_survives_a_new_graph_object(graph: AgentGraph, tmp_path: Path) -> None:
    """Proves the job is genuinely on disk, not merely in memory, status type included."""
    started = graph.start("a feature")

    reread = AgentGraph("bdd", {}, start="write", state_dir=tmp_path / ".rooster").load(started.slug)

    assert reread.slug == started.slug
    assert reread.next_step == "test"
    assert reread.status is JobStatus.WAITING


def test_usage_is_totalled_across_agents(graph: AgentGraph) -> None:
    """Proves the job accumulates what every agent spent, not just the last one."""
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


def test_jobs_sharing_agents_do_not_count_each_others_spend(graph: AgentGraph) -> None:
    """Proves each job counts only what was spent on it, though the agents are shared."""
    first = graph.start("a feature")
    second = graph.start("another feature")

    assert first.tokens == 100
    assert second.tokens == 100


def test_a_halted_step_still_counts_its_spend(tmp_path: Path) -> None:
    """Proves a step that spends and then raises leaves its spend on the saved job."""
    agents = {"writer": FakeAgent("writer", tmp_path)}
    g = AgentGraph("x", agents, start="boom", state_dir=tmp_path / ".rooster")

    @g.step("boom", next=(None,))
    def boom(job: Job) -> None:
        agents["writer"].run("anything")
        raise ValueError("the agent wrote nonsense")

    with pytest.raises(ValueError):
        g.start("a feature")

    saved = g.jobs()[0]
    assert saved.status == JobStatus.HALTED
    assert saved.cost_usd == pytest.approx(0.01)


def test_slugs_do_not_collide(graph: AgentGraph) -> None:
    """Proves two jobs started from the same words get separate files."""
    first = graph.start("the same feature")
    second = graph.start("the same feature")

    assert first.slug == "the-same-feature"
    assert second.slug == "the-same-feature-2"
    assert len(graph.jobs()) == 2


def test_a_step_that_raises_halts_and_is_saved(tmp_path: Path) -> None:
    """Proves a crash inside a step is recorded on disk before the exception escapes."""
    g = AgentGraph("x", {}, start="boom", state_dir=tmp_path / ".rooster")

    @g.step("boom", next=(None,))
    def boom(job: Job) -> None:
        raise ValueError("the agent wrote nonsense")

    with pytest.raises(ValueError):
        g.start("a feature")

    saved = g.jobs()[0]
    assert saved.status == JobStatus.HALTED
    assert "ValueError" in saved.halt_reason


def test_an_undeclared_next_step_halts(tmp_path: Path) -> None:
    """Proves a step returning a name it did not declare stops the job and says so."""
    g = AgentGraph("x", {}, start="one", state_dir=tmp_path / ".rooster")

    @g.step("one", next=("two",))
    def one(job: Job) -> str:
        return "tow"  # typo for "two"

    @g.step("two", next=(None,))
    def two(job: Job) -> None:
        return None

    job = g.start("a feature")

    assert job.status == JobStatus.HALTED
    assert "'tow'" in job.halt_reason
    assert "'two'" in job.halt_reason


def test_a_declared_step_that_does_not_exist_is_caught_before_any_agent_runs(
    tmp_path: Path,
) -> None:
    """Proves a typo in a 'next' declaration fails at start, not halfway through a job."""
    g = AgentGraph("x", {}, start="one", state_dir=tmp_path / ".rooster")
    ran: list[str] = []

    @g.step("one", next=("tow",))
    def one(job: Job) -> str:
        ran.append("one")
        return "tow"

    with pytest.raises(GraphError, match="declares \\['tow'\\]"):
        g.start("a feature")
    assert ran == []


def test_a_runaway_loop_is_capped(tmp_path: Path) -> None:
    """Proves a cycle stops loudly instead of running forever."""
    g = AgentGraph("x", {}, start="again", state_dir=tmp_path / ".rooster", max_steps=5)

    @g.step("again", next=("again",))
    def again(job: Job) -> str:
        job.data["rounds"] = job.data.get("rounds", 0) + 1
        return "again"

    job = g.start("a feature")

    assert job.status == JobStatus.HALTED
    assert "5 steps" in job.halt_reason
    assert job.data["rounds"] == 5


def test_an_unregistered_start_is_caught_before_any_agent_runs(tmp_path: Path) -> None:
    """Proves a misdeclared graph fails at once, not halfway through a job."""
    g = AgentGraph("x", {}, start="missing", state_dir=tmp_path / ".rooster")

    @g.step("present", next=(None,))
    def present(job: Job) -> None:
        return None

    with pytest.raises(GraphError, match="start step 'missing' is not registered"):
        g.start("a feature")


def test_a_duplicate_step_name_is_rejected(tmp_path: Path) -> None:
    """Proves two steps cannot quietly claim the same name."""
    g = AgentGraph("x", {}, start="one", state_dir=tmp_path / ".rooster")

    @g.step("one", next=(None,))
    def one(job: Job) -> None:
        return None

    with pytest.raises(GraphError, match="already registered"):

        @g.step("one", next=(None,))
        def one_again(job: Job) -> None:
            return None


def test_resuming_a_finished_job_is_refused(graph: AgentGraph) -> None:
    """Proves a completed job cannot be restarted by accident."""
    started = graph.start("a feature")
    graph.resume(started.slug)

    with pytest.raises(GraphError, match="already done"):
        graph.resume(started.slug)


def test_an_unknown_job_names_the_ones_that_exist(graph: AgentGraph) -> None:
    """Proves a mistyped slug gets a useful error rather than a stack trace."""
    graph.start("a real feature")

    with pytest.raises(GraphError, match="no job named 'nope'"):
        graph.resume("nope")


def test_the_diagram_shows_every_step_arrow_and_gate(graph: AgentGraph) -> None:
    """Proves diagram() draws the start, each declared arrow, the gate and the finish."""
    assert graph.diagram().splitlines() == [
        "flowchart TD",
        "    start((start)) --> s0",
        '    s0["write (gate)"]',
        '    s1["test"]',
        "    s0 --> s1",
        "    s1 --> finish((finish))",
    ]


def test_the_diagram_refuses_an_invalid_graph(tmp_path: Path) -> None:
    """Proves diagram() checks the graph first, so it never draws an arrow to nowhere."""
    g = AgentGraph("x", {}, start="one", state_dir=tmp_path / ".rooster")

    @g.step("one", next=("missing",))
    def one(job: Job) -> str:
        return "missing"

    with pytest.raises(GraphError, match="not registered"):
        g.diagram()
