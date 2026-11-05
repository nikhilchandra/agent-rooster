"""Runs one job through a graph of steps, saving its place so it can stop for a person and
resume later.

A graph is a set of named steps. A step is a plain function that takes the Job, does its
work, usually by calling an agent or running a check, and returns the name of the next step.
Returning None finishes the job. Each step declares every step it may go to with 'next', so
the graph can be checked before it runs and drawn with diagram().

A loop is a step that returns an earlier step's name. A gate is a step declared with
gate=True, after which the job pauses until resume() is called, possibly from a new process.
A deterministic check is ordinary Python inside a step, with no agent involved.

A graph describes one job, such as one refactor step from plan to commit. Repeating jobs,
such as working through a list of problems, is a loop in the caller's own code that starts
a new job each time.

Example, a refactor loop that plans, implements, tests and reviews::

    graph = AgentGraph(
        "refactor",
        {"planner": planner, "implementer": implementer, "reviewer": reviewer},
        start="plan",
        state_dir=cwd / ".rooster",
    )

    @graph.step("plan", next=("implement",), gate=True)
    def plan(job: Job) -> str:
        job.data["plan"] = planner.run(job.prompt)["data"].model_dump()
        return "implement"

    @graph.step("implement", next=("test",))
    def implement(job: Job) -> str:
        implementer.run(prompt_from(job.data))
        return "test"

    @graph.step("test", next=("review", "implement"))
    def test(job: Job) -> str:
        passed = subprocess.run(["uv", "run", "pytest"]).returncode == 0
        return "review" if passed else "implement"

    @graph.step("review", next=("implement", None))
    def review(job: Job) -> str | None:
        verdict = reviewer.run(diff_prompt())["data"]
        if verdict.accepted:
            return None
        job.data["rejections"] = job.data.get("rejections", 0) + 1
        return "implement" if job.data["rejections"] < 3 else None

    job = graph.start("Fix the next problem in problems.md")   # stops after "plan"
    job = graph.resume(job.slug)                                 # runs to the end
    print(graph.diagram())
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from rooster.agent import Agent


class GraphError(RuntimeError):
    """Signals a problem with the graph itself, not with an agent's answer."""


class JobStatus(StrEnum):
    """Names the states a job can be in."""

    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    HALTED = "halted"


@dataclass
class Job:
    """Holds one trip through a graph, from start to finish, and is saved after every step."""

    # A short name made from the prompt, such as "add-a-login-page". It is also the job's
    # file name.
    slug: str
    # The step that runs next, or None once the job has ended.
    next_step: str | None
    # The task that the job was started with. The steps turn it into messages for their
    # agents, so it is not any one agent's profile.
    prompt: str
    # The session id of each agent that remembers, keyed by the agent's name, so that a
    # resumed job's agents rejoin their conversations.
    sessions: dict[str, str] = field(default_factory=dict)
    # Anything the steps keep between steps. It is saved as JSON, so a schema instance is
    # stored with model_dump().
    data: dict = field(default_factory=dict)
    # What the job has spent so far, in dollars and in tokens.
    cost_usd: float = 0.0
    tokens: int = 0
    # Whether the job is running, waiting at a gate, done or halted.
    status: JobStatus = JobStatus.RUNNING
    # The reason the job halted, or None when it has not halted.
    halt_reason: str | None = None
    # When the job started and when it was last saved, as ISO timestamps in UTC.
    started_at: str = ""
    updated_at: str = ""

    @property
    def waiting(self) -> bool:
        """Answers whether this job is paused at a gate, waiting for a person."""
        return self.status == JobStatus.WAITING

    @property
    def finished(self) -> bool:
        """Answers whether this job will not advance again."""
        return self.status in {JobStatus.DONE, JobStatus.HALTED}


# a step takes the job and returns the name of the next step, or None to finish
Step = Callable[[Job], "str | None"]


def _slugify(text: str) -> str:
    """Returns a short name for a job that is safe to use as a file name.

    Args:
        text: The job's prompt.

    Returns:
        The prompt in lowercase, with every run of other characters turned into one hyphen
        and cut to 48 characters, or "job" when nothing is left.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return cleaned[:48].rstrip("-") or "job"


def _now() -> str:
    """Returns the current time as an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AgentGraph:
    """Declares a set of steps over a group of agents, and runs jobs through them."""

    def __init__(
        self,
        name: str,
        agents: dict[str, Agent],
        start: str,
        *,
        state_dir: Path,
        max_steps: int = 100,
    ) -> None:
        """Registers the agents and decides where this graph's jobs are stored.

        Args:
            name: The graph's name. Its jobs are stored under 'state_dir' in a folder of
                this name.
            agents: The agents that the steps call, keyed by the name a job saves their
                sessions under.
            start: The name of the step that every job begins with.
            state_dir: The folder of the project using rooster, such as <cwd>/.rooster.
            max_steps: The most steps that one start or resume may take before the job
                halts.
        """
        self.name = name
        self.agents = agents
        self.start_step = start
        self.state_dir = Path(state_dir) / name
        self.max_steps = max_steps

        # Each step's function, the steps it declares it may go to, and the steps that are
        # gates.
        self._steps: dict[str, Step] = {}
        self._next: dict[str, tuple[str | None, ...]] = {}
        self._gates: set[str] = set()
        # The agents' combined cost and tokens at the last count, so a job is charged only
        # for what was spent while it was advancing.
        self._counted = (0.0, 0)

    # ------------------------------------------------------------------ declaring

    def step(
        self, name: str, *, next: tuple[str | None, ...], gate: bool = False
    ) -> Callable[[Step], Step]:
        """Returns a decorator that registers a function as a named step.

        Args:
            name: The step's name, which other steps return to send the job here.
            next: Every step that this step may return, with None for finishing the job.
            gate: True pauses the job after this step until resume() is called.

        Returns:
            A decorator that registers the function and returns it unchanged.

        Raises:
            GraphError: A step of this name is already registered.
        """

        def register(fn: Step) -> Step:
            if name in self._steps:
                raise GraphError(f"graph {self.name!r}: step {name!r} is already registered")
            self._steps[name] = fn
            self._next[name] = tuple(next)
            if gate:
                self._gates.add(name)
            return fn

        return register

    def diagram(self) -> str:
        """Returns the graph as a Mermaid flowchart.

        GitHub, VS Code and Claude artifacts draw Mermaid text as a picture, and it stays
        readable as plain text.

        Returns:
            The flowchart's text, with each gate labeled "(gate)".

        Raises:
            GraphError: The graph is not valid, as _validate() checks.
        """
        self._validate()
        ids = {name: f"s{index}" for index, name in enumerate(self._steps)}
        lines = ["flowchart TD", f"    start((start)) --> {ids[self.start_step]}"]
        for name, node in ids.items():
            label = f"{name} (gate)" if name in self._gates else name
            lines.append(f'    {node}["{label}"]')
        for name, targets in self._next.items():
            for target in targets:
                lines.append(f"    {ids[name]} --> {ids[target] if target else 'finish((finish))'}")
        return "\n".join(lines)

    def _validate(self) -> None:
        """Checks that the graph can run, before any agent is called.

        Raises:
            GraphError: No step is registered, the start step is not registered, or a step
                declares a next step that is not registered.
        """
        if not self._steps:
            raise GraphError(f"graph {self.name!r}: no steps registered")
        if self.start_step not in self._steps:
            raise GraphError(
                f"graph {self.name!r}: start step {self.start_step!r} is not registered. "
                f"known steps: {sorted(self._steps)}"
            )
        for name, targets in self._next.items():
            unknown = [target for target in targets if target is not None and target not in self._steps]
            if unknown:
                raise GraphError(
                    f"graph {self.name!r}: step {name!r} declares {unknown} as next, "
                    f"which are not registered. known steps: {sorted(self._steps)}"
                )

    # ------------------------------------------------------------------- running

    def start(self, prompt: str, slug: str | None = None) -> Job:
        """Begins a new job and advances it until it finishes, halts or reaches a gate.

        Args:
            prompt: The task that the job is started with.
            slug: The job's name. None makes one from 'prompt'. A counter is added when the
                name is already taken.

        Returns:
            The job as it stands when it stopped advancing.
        """
        self._validate()
        job = Job(
            slug=self._unique_slug(slug or _slugify(prompt)),
            next_step=self.start_step,
            prompt=prompt,
            started_at=_now(),
        )
        self._save(job)
        return self._advance(job)

    def resume(self, slug: str, data: dict | None = None) -> Job:
        """Reloads a paused job, hands its sessions back to the agents, and continues it.

        Args:
            slug: The job's name.
            data: Decisions made at the gate, such as an owner's answers. Each key is
                written into 'job.data' before the next step runs, replacing any value
                already under that key. None adds nothing.

        Returns:
            The job as it stands when it stopped advancing.

        Raises:
            GraphError: No job has this name, or the job has already finished.
        """
        self._validate()
        job = self.load(slug)
        if job.finished:
            raise GraphError(f"job {slug!r} is already {job.status}, so there is nothing to resume")
        job.status = JobStatus.RUNNING
        # merge the gate's decisions into the job's data
        if data:
            job.data.update(data)
        self._restore_sessions(job)
        return self._advance(job)

    def load(self, slug: str) -> Job:
        """Reads a job from disk without advancing it.

        Args:
            slug: The job's name.

        Returns:
            The job as it was last saved.

        Raises:
            GraphError: No job has this name, or its file cannot be read.
        """
        path = self._path(slug)
        if not path.is_file():
            known = [p.stem for p in self.state_dir.glob("*.json")] if self.state_dir.is_dir() else []
            raise GraphError(f"no job named {slug!r} in {self.state_dir}. known jobs: {known}")
        return self._read(path)

    def jobs(self) -> list[Job]:
        """Returns every job this graph has stored, most recently saved first."""
        if not self.state_dir.is_dir():
            return []
        found = [self._read(path) for path in self.state_dir.glob("*.json")]
        return sorted(found, key=lambda job: job.updated_at, reverse=True)

    # ------------------------------------------------------------------ internals

    def _advance(self, job: Job) -> Job:
        """Calls steps in turn until the job finishes, halts or pauses at a gate.

        Args:
            job: The job to advance, with 'next_step' naming where it continues.

        Returns:
            The job as it stands when it stopped advancing.
        """
        self._counted = self._usage()
        for _ in range(self.max_steps):
            if job.next_step is None:
                return self._finish(job, JobStatus.DONE)

            step = self._steps.get(job.next_step)
            if step is None:
                return self._finish(
                    job, JobStatus.HALTED, f"step {job.next_step!r} is not registered"
                )

            here = job.next_step
            try:
                after = step(job)
            except Exception as exc:
                # the job is saved before the exception escapes, so nothing is lost
                self._finish(job, JobStatus.HALTED, f"step {here!r} raised {type(exc).__name__}: {exc}")
                raise

            if after not in self._next[here]:
                return self._finish(
                    job,
                    JobStatus.HALTED,
                    f"step {here!r} returned {after!r}, but declares only {list(self._next[here])}",
                )

            job.next_step = after
            self._collect(job)
            self._save(job)

            if here in self._gates:
                job.status = JobStatus.WAITING
                self._save(job)
                return job

        return self._finish(job, JobStatus.HALTED, f"stopped after {self.max_steps} steps")

    def _finish(self, job: Job, status: JobStatus, halt_reason: str | None = None) -> Job:
        """Marks a job as ended and saves it.

        Args:
            job: The job that has ended.
            status: JobStatus.DONE or JobStatus.HALTED.
            halt_reason: The reason the job halted, or None when it finished normally.

        Returns:
            The same job, now saved.
        """
        job.status = status
        job.halt_reason = halt_reason
        job.next_step = None
        self._collect(job)
        self._save(job)
        return job

    def _collect(self, job: Job) -> None:
        """Copies each agent's session id and new spending onto the job.

        Args:
            job: The job to update.
        """
        for name, agent in self.agents.items():
            if agent.session_id:
                job.sessions[name] = agent.session_id
        # add only the spending since the last count, since agents may be new or shared by jobs
        cost, tokens = self._usage()
        job.cost_usd += cost - self._counted[0]
        job.tokens += tokens - self._counted[1]
        self._counted = (cost, tokens)

    def _usage(self) -> tuple[float, int]:
        """Returns the dollars and tokens that the agents have spent so far, combined."""
        return (
            sum(a.total_cost_usd for a in self.agents.values()),
            sum(a.total_tokens for a in self.agents.values()),
        )

    def _restore_sessions(self, job: Job) -> None:
        """Hands each saved session id back to its agent, so the agent rejoins its conversation.

        Args:
            job: The job whose saved sessions are restored.
        """
        for name, session_id in job.sessions.items():
            agent = self.agents.get(name)
            if agent is not None:
                agent.session_id = session_id

    # ------------------------------------------------------------------- storage

    def _path(self, slug: str) -> Path:
        """Returns the file that the job named 'slug' is stored in."""
        return self.state_dir / f"{slug}.json"

    def _unique_slug(self, base: str) -> str:
        """Returns 'base', with a counter added when a job of that name already exists."""
        if not self._path(base).exists():
            return base
        n = 2
        while self._path(f"{base}-{n}").exists():
            n += 1
        return f"{base}-{n}"

    def _save(self, job: Job) -> None:
        """Writes a job to disk, replacing its file in one step.

        Args:
            job: The job to save. Its 'updated_at' is set to now.
        """
        job.updated_at = _now()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        target = self._path(job.slug)
        # write beside the target and rename, so a crash mid-write cannot truncate the state
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        temp.replace(target)

    def _read(self, path: Path) -> Job:
        """Turns one saved file back into a Job.

        Args:
            path: The job's file.

        Returns:
            The job as it was saved.

        Raises:
            GraphError: The file is not a saved job.
        """
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["status"] = JobStatus(saved["status"])
            return Job(**saved)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise GraphError(f"the job file {path} is unreadable: {exc}") from exc
