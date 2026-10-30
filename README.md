# agent-rooster

A harness for running Claude Code agents from Python.

agent-rooster lets you define an agent, run it, and get its answer back as data your
script can act on. It also runs several agents through a graph of steps, saving each job
to disk so that it can pause for a person and resume later, even from a new process.

It knows nothing about code, tests or git, and ships no roles or workflows of its own.
Tools built on it bring that knowledge. ratchet, a coding tool with `refactor`, `feature`
and `debug` commands, is the first one planned.

Agents run through Anthropic's [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python),
which drives the `claude` CLI. So an agent can use Claude Code's tools and file editing,
and it runs against your Claude plan.

## Install

agent-rooster is a library. Add it to another project's `pyproject.toml` as an editable
path, so changes here show up there at once:

```toml
[project]
dependencies = ["agent-rooster"]

[tool.uv.sources]
agent-rooster = { path = "../agent-rooster", editable = true }
```

## Agents

An agent is a profile, which is a markdown file holding its standing instructions, plus
the settings below.

```python
from pathlib import Path
from pydantic import BaseModel
from rooster import Agent, Command, Model, Tool

class Verdict(BaseModel):
    accepted: bool
    reason: str

reviewer = Agent(
    "reviewer",
    Path("profiles/reviewer.md"),
    cwd=Path("."),
    log_dir=Path(".rooster/logs/reviewer"),
    model=Model.OPUS,
    tools=Tool.INSPECT,                       # read-only: Read, Glob and Grep
    commands=Command.PYTEST | Command.GIT_READ,
    schema=Verdict,
    max_work_usd=3.0,
    timeout_s=1800,
)

result = reviewer.run("Review the diff in the working tree.")
if result["ok"]:
    print(result["data"].accepted, result["data"].reason)
else:
    print(result["failure"])
```

The settings, briefly:

| Setting | What it does |
|---|---|
| `cwd`, `log_dir` | Where the agent works, and where it writes one log file per turn. Both are required. A log is written line by line, so you can watch a long turn live. |
| `remembers` | False (the default) starts every turn fresh, so the prompt must carry everything the agent needs. True continues one conversation across all of the agent's turns. |
| `model`, `effort` | Which model, and how much it thinks. |
| `tools` | The built-in tools the agent may use, such as `Tool.INSPECT` (read only) or `Tool.AUTHOR` (read and edit). |
| `commands` | The shell commands it may run, such as `Command.PYTEST`, `Command.GIT_MV`, `Command.GIT_RM` and `Command.GIT_READ`. `Tool.BASH` allows every command instead. |
| `editable`, `frozen` | Path patterns it may change, such as `"src/**"`, and patterns it may never change, such as `"**/*.feature"`. Claude Code enforces both. |
| `schema` | A pydantic class that the reply must fit. The reply arrives as an instance of it in `result["data"]`, and a reply that does not fit makes `result["ok"]` False. |
| `max_work_usd`, `timeout_s` | The most one turn may spend, in dollars, and the longest it may take, in seconds. |

`run()` waits for the turn to finish. `arun()` is the same for async code, which lets
several agents work at once:

```python
first, second = await asyncio.gather(reviewer_a.arun(prompt), reviewer_b.arun(prompt))
```

One agent must not run two turns at once. Use separate agents instead.

The `Agent` class docstring lists every key of the result dict.

## Graphs and jobs

A graph is a set of named steps. A step is a plain function that takes the `Job`, does its
work, and returns the name of the next step, or None to finish. Each step declares every
step it may go to with `next=`, so a typo is caught before any agent runs.

- **A loop** is a step that returns an earlier step's name.
- **A gate** is a step declared with `gate=True`. The job pauses after it until you call
  `resume()`.
- **A check** is ordinary Python inside a step, such as running the tests, with no agent
  involved.

```python
from rooster import AgentGraph, Job

graph = AgentGraph("review", {"writer": writer, "reviewer": reviewer},
                   start="write", state_dir=Path(".rooster"))

@graph.step("write", next=("test",))
def write(job: Job) -> str:
    writer.run(job.prompt)
    return "test"

@graph.step("test", next=("review", "write"))
def test(job: Job) -> str:
    passed = subprocess.run(["uv", "run", "pytest"]).returncode == 0
    return "review" if passed else "write"

@graph.step("review", next=("write", None), gate=True)
def review(job: Job) -> str | None:
    verdict = reviewer.run("Review the diff.")["data"]
    return None if verdict.accepted else "write"

job = graph.start("Add a --verbose flag to the CLI")
if job.waiting:
    job = graph.resume(job.slug)
```

A job is one trip through the graph. Each job is saved as a JSON file under
`state_dir/<graph name>/`, holding its next step, its status, what it has cost, the
session ids of agents that remember, and `job.data`, where steps keep anything they need
later. To work through many tasks, such as a list of problems, write a loop in your own
code that starts one job per task.

`graph.diagram()` returns the graph as a Mermaid flowchart, which GitHub and VS Code draw
as a picture:

```
flowchart TD
    start((start)) --> s0
    s0["write"]
    s1["test"]
    s2["review (gate)"]
    s0 --> s1
    s1 --> s2
    s1 --> s0
    s2 --> s0
    s2 --> finish((finish))
```

## Tests

```bash
uv run pytest -m "not live"   # free, no Claude calls
uv run pytest                 # also runs the live tests, which spend a little of your plan
```

## Status

Working now: agents, parallel agents, permissions, schemas, live logs, graphs with
loops, gates and checks, and diagrams.

Planned next, in order:
1. A halted job keeps its place, so it can resume from the step that failed.
2. Several jobs at once, with a lock per job and a progress log.
3. A dollar cap per job, and ways to stop a job between steps.
4. Waiting out the plan's usage limits instead of failing.
5. Checking an agent's permissions before a long job, without calling Claude.
6. Building blocks for a project's own command line.
