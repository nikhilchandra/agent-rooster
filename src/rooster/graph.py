"""Runs a sequence of agents, saving its place so a run can stop for a human and resume."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from claude_agent_sdk import tag_session

from rooster.agent import Agent


class GraphError(RuntimeError):
    """Signals a problem with the graph itself, not with an agent's answer."""


# a step takes the run and returns the name of the next step, or None to finish
Step = Callable[["Run"], "str | None"]


@dataclass
class Run:
    """Holds one execution of a graph, and is saved to disk between steps."""

    slug: str
    cursor: str | None
    prompt: str
    sessions: dict[str, str] = field(default_factory=dict)
    data: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    tokens: int = 0
    revisions: int = 0
    status: str = "running"
    failure: str | None = None
    started_at: str = ""
    updated_at: str = ""

    @property
    def waiting(self) -> bool:
        """Says whether this run is parked at a gate waiting for the human."""
        return self.status == "awaiting_human"

    @property
    def finished(self) -> bool:
        """Says whether this run will not advance again."""
        return self.status in {"done", "halted"}


def _slugify(text: str) -> str:
    """Turns a prompt into a short filesystem-safe name."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (cleaned[:48].rstrip("-") or "run")


def _now() -> str:
    """Returns an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AgentGraph:
    """Declares an ordered set of steps over a group of agents, and runs them."""

    def __init__(
        self,
        name: str,
        agents: dict[str, Agent],
        start: str,
        *,
        state_dir: Path = Path(".rooster"),
        max_steps: int = 100,
    ) -> None:
        """Registers the agents and decides where this graph's runs are stored."""
        self.name = name
        self.agents = agents
        self.start_step = start
        self.state_dir = Path(state_dir) / name
        self.max_steps = max_steps

        self._steps: dict[str, Step] = {}
        self._gates: set[str] = set()
        # The agents' combined cost and tokens at the last count, so a run is charged only
        # for what was spent while it was advancing.
        self._counted = (0.0, 0)

    # ------------------------------------------------------------------ declaring

    def step(self, name: str, *, gate: bool = False) -> Callable[[Step], Step]:
        """Registers a function as a named step, optionally one that stops for the human."""

        def register(fn: Step) -> Step:
            if name in self._steps:
                raise GraphError(f"graph {self.name!r}: step {name!r} is already registered")
            self._steps[name] = fn
            if gate:
                self._gates.add(name)
            return fn

        return register

    def validate(self) -> None:
        """Checks the graph is runnable before any agent is called."""
        if not self._steps:
            raise GraphError(f"graph {self.name!r}: no steps registered")
        if self.start_step not in self._steps:
            raise GraphError(
                f"graph {self.name!r}: start step {self.start_step!r} is not registered. "
                f"known steps: {sorted(self._steps)}"
            )

    # ------------------------------------------------------------------- running

    def start(self, prompt: str, slug: str | None = None) -> Run:
        """Begins a new run and advances it until it finishes or reaches a gate."""
        self.validate()
        run = Run(
            slug=self._unique_slug(slug or _slugify(prompt)),
            cursor=self.start_step,
            prompt=prompt,
            started_at=_now(),
        )
        self._save(run)
        return self._advance(run)

    def resume(self, slug: str) -> Run:
        """Reloads a parked run, hands its sessions back to the agents, and continues."""
        self.validate()
        run = self._load(slug)
        if run.finished:
            raise GraphError(f"run {slug!r} is already {run.status}; nothing to resume")
        run.status = "running"
        self._restore_sessions(run)
        return self._advance(run)

    def status(self, slug: str) -> Run:
        """Returns a run's saved state without advancing it."""
        return self._load(slug)

    def runs(self) -> list[Run]:
        """Lists every run this graph has in the current project, newest first."""
        if not self.state_dir.is_dir():
            return []
        found = [self._read(path) for path in self.state_dir.glob("*.json")]
        return sorted(found, key=lambda r: r.updated_at, reverse=True)

    # ------------------------------------------------------------------ internals

    def _advance(self, run: Run) -> Run:
        """Calls steps in turn until the run finishes, halts, or parks at a gate."""
        self._counted = self._usage()
        for _ in range(self.max_steps):
            if run.cursor is None:
                return self._finish(run, "done")

            step = self._steps.get(run.cursor)
            if step is None:
                return self._finish(
                    run, "halted", f"step {run.cursor!r} is not registered"
                )

            here = run.cursor
            try:
                nxt = step(run)
            except Exception as exc:
                # the run is saved before the exception escapes, so nothing is lost
                self._finish(run, "halted", f"step {here!r} raised {type(exc).__name__}: {exc}")
                raise

            run.cursor = nxt
            self._collect(run)
            self._save(run)

            if here in self._gates:
                run.status = "awaiting_human"
                self._save(run)
                return run

        return self._finish(run, "halted", f"stopped after {self.max_steps} steps")

    def _finish(self, run: Run, status: str, failure: str | None = None) -> Run:
        """Marks a run as ended and writes it out."""
        run.status = status
        run.failure = failure
        run.cursor = None
        self._collect(run)
        self._save(run)
        return run

    def _collect(self, run: Run) -> None:
        """Copies each agent's session id and usage onto the run, so a resume can restore them."""
        for name, agent in self.agents.items():
            if agent.session_id:
                run.sessions[name] = agent.session_id
                # tag the session with its graph and run, so it can be traced back from
                # Claude Code's session list. a session that cannot be tagged is skipped,
                # since the tag is only for a human browsing sessions
                try:
                    tag_session(
                        agent.session_id,
                        f"rooster:{self.name}/{run.slug}",
                        directory=str(agent.cwd),
                    )
                except (ValueError, FileNotFoundError):
                    pass
        # add only the spend since the last count, since agents may be new or shared by runs
        cost, tokens = self._usage()
        run.cost_usd += cost - self._counted[0]
        run.tokens += tokens - self._counted[1]
        self._counted = (cost, tokens)

    def _usage(self) -> tuple[float, int]:
        """Returns the cost and tokens that the agents have spent so far, combined."""
        return (
            sum(a.total_cost_usd for a in self.agents.values()),
            sum(a.total_tokens for a in self.agents.values()),
        )

    def _restore_sessions(self, run: Run) -> None:
        """Puts saved session ids back on the agents so they rejoin their conversations."""
        for name, session_id in run.sessions.items():
            agent = self.agents.get(name)
            if agent is None:
                continue
            # an agent holding an id always resumes, so this is the whole restoration
            agent.session_id = session_id

    # ------------------------------------------------------------------- storage

    def _path(self, slug: str) -> Path:
        """Gives the file a run is stored in."""
        return self.state_dir / f"{slug}.json"

    def _unique_slug(self, base: str) -> str:
        """Appends a counter if a run by this name already exists."""
        if not self._path(base).exists():
            return base
        n = 2
        while self._path(f"{base}-{n}").exists():
            n += 1
        return f"{base}-{n}"

    def _save(self, run: Run) -> None:
        """Writes a run to disk, replacing the file atomically."""
        run.updated_at = _now()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        target = self._path(run.slug)
        # write beside the target and rename, so a crash mid-write cannot truncate the state
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
        temp.replace(target)

    def _load(self, slug: str) -> Run:
        """Reads a run from disk."""
        path = self._path(slug)
        if not path.is_file():
            known = [p.stem for p in self.state_dir.glob("*.json")] if self.state_dir.is_dir() else []
            raise GraphError(f"no run named {slug!r} in {self.state_dir}. known runs: {known}")
        return self._read(path)

    def _read(self, path: Path) -> Run:
        """Turns one state file into a Run."""
        try:
            return Run(**json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            raise GraphError(f"the run file {path} is unreadable: {exc}") from exc
