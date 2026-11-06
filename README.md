# agent-rooster

Build workflows of AI agents in Python that run on your Claude subscription, not on a
pay-per-token API.

## What it is

agent-rooster is a small Python library. It lets you:

- **Run a Claude agent from Python** and get its answer back as data your code can use.
- **Connect several steps into a workflow.** A step can call an agent, run ordinary code
  (such as your tests), or both. Your code decides which step comes next, not the AI.
- **Pause a workflow for a person** and pick it up later, even after closing the program.
- **Keep each agent inside strict limits**: which files it may read, which it may change, and
  which commands it may run.

Each agent is a real Claude Code agent. It can read, search and edit files and run commands,
using Claude Code's own tools, so you do not have to write those tools yourself.

## Why it exists

Popular frameworks for agent workflows (e.g., LangGraph) typically call an AI model through an API. You pay for every token. If you already pay for a Claude Pro or Max plan, that may feel like paying twice.

agent-rooster runs every agent through Claude Code using your own plan. Your workflows use your plan's usage limits instead of a token bill.

It uses Anthropic's Claude Agent SDK to configure and run agents, and checks each agent's answer against a shape that you define. Its AgentGraph runs complicated workflows in which your code, not the AI, decides what happens next.

[code-ratchet](https://github.com/nikhilchandra/code-ratchet), a tool that audits and
refactors Python projects with a team of agents, is built on agent-rooster.

## What you need

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/), a Python package manager
- [Claude Code](https://docs.claude.com/en/docs/claude-code), installed and signed in with
  your Claude plan

## Install

```bash
git clone https://github.com/nikhilchandra/agent-rooster
cd agent-rooster
uv sync
uv run pytest        # runs the tests, with no calls to Claude
```

To use it in your own project, add it as a dependency. You install `agent-rooster`, and you
import `rooster`.

```toml
[project]
dependencies = ["agent-rooster"]

[tool.uv.sources]
agent-rooster = { git = "https://github.com/nikhilchandra/agent-rooster" }
```

## Try it

[examples/review](examples/review) holds a working example: a tiny project with one function,
and a writer and a reviewer, each with its own instructions in a markdown file. The writer
adds a function with a test, the tests run, and the reviewer judges the change. It pauses for
you before it starts, so you can look at the starting code, and after each review.

```bash
uv run python examples/review/run.py
```

It copies the project into `examples/review/work/` first, so you can see what the writer
changed. The writer edits only that copy, and git ignores it and the run's logs, so a run
leaves no changes in the repository. A run uses a few cents of your plan's usage, measured at
API prices.

The two sections below walk through the same pieces.

## Your first agent

An agent is a set of instructions in a markdown file, plus a few settings.

```python
from pathlib import Path
from pydantic import BaseModel
from rooster import Agent, Command, Model, Tool

# the shape the agent's answer must have
class Verdict(BaseModel):
    accepted: bool
    reason: str

reviewer = Agent(
    "reviewer",
    Path("profiles/reviewer.md"),        # the agent's instructions (this file must exist)
    cwd=Path("."),                       # the folder it works in
    log_dir=Path(".rooster/logs"),       # one log file per turn
    model=Model.OPUS,
    tools=Tool.INSPECT,                  # it may read files, but not change them
    commands=Command.PYTEST | Command.GIT_READ,
    schema=Verdict,
    max_work_usd=3.0,                    # usage cap: stop the turn once it would have cost $3 at API prices
)

result = reviewer.run("Review the changes in this folder.")
if result["ok"]:
    print(result["data"].accepted, result["data"].reason)
else:
    print("The turn failed:", result["failure"])
```

**Note**: On a subscription plan, nothing is billed. The cost is Claude Code's estimate at API 
prices, which rooster uses to cap usage and to track what each job used.

If the agent's answer does not fit `Verdict`, `result["ok"]` is False and you get the reason
instead.

## Your first workflow

A workflow is a set of named steps. Each step is a plain function that returns the name of
the next step, or `None` to finish. Every step lists where it may go next, so a typo is
caught before any agent runs.

```python
import subprocess
from pathlib import Path

from rooster import AgentGraph, Job

# writer is an Agent defined like reviewer above, with remembers=True and no schema
graph = AgentGraph("review", {"writer": writer, "reviewer": reviewer},
                   start="write", state_dir=Path(".rooster"))

@graph.step("write", next=("test",))
def write(job: Job) -> str:
    feedback = job.data.pop("feedback", "")
    writer.run(f"{job.prompt}\n\n{feedback}".strip())   # an agent step
    return "test"

@graph.step("test", next=("review", "write"))
def test(job: Job) -> str:
    tests = subprocess.run(["uv", "run", "pytest"], capture_output=True, text=True)
    if tests.returncode == 0:
        return "review"                                  # a code step, with no agent
    job.data["feedback"] = f"The tests failed:\n{tests.stdout[-2000:]}"
    return "write"

@graph.step("review", next=("write", None), gate=True)
def review(job: Job) -> str | None:
    result = reviewer.run("Review the changes.")
    if not result["ok"]:
        raise RuntimeError(result["failure"])            # halts the job with this reason
    if result["data"].accepted:
        return None
    job.data["feedback"] = f"The reviewer rejected the changes: {result['data'].reason}"
    return "write"

job = graph.start("Add a --verbose flag to the command line")
while job.waiting:            # paused after each review, so a person can look
    job = graph.resume(job.slug)
```

A few things to know:

- **A loop** is just a step that returns an earlier step's name.
- **A gate** (`gate=True`) pauses the job after that step. `resume()` continues it, and
  `resume(slug, data={...})` hands it a person's decision.
- **Each job is saved** as a JSON file after every step, with its next step, its cost and
  any data the steps keep. That is what lets a paused job continue later, in a new process.
- **A job that fails is marked halted.** It does not pick up where it failed. Start a new job
  instead.
- `graph.diagram()` draws the workflow as a Mermaid flowchart, which GitHub shows as a
  picture.

## Keeping agents in bounds

You decide what each agent may touch. Claude Code and agent-rooster enforce it, so the agent
cannot simply ignore it.

| Setting | What it does |
|---|---|
| `tools` | Which of Claude Code's tools the agent has, such as read-only or read-and-edit |
| `commands` | Which shell commands it may run, such as `pytest` or read-only git |
| `editable` | The files it may change, such as `"src/**"` |
| `frozen` | Files it may never change, such as tests written before the code |
| `hidden` | Files and folders it may never read |
| `confined_to` | A folder its file tools may not leave. Each path is checked by where it really points, so links, `..` and Windows short names cannot get around it |
| `forbidden_options` | Command options that are refused before the command runs, such as git's `--no-index`, which can read files outside the project |
| `setting_sources` | Which Claude Code settings files the agent loads. An empty list loads none, so a settings file cannot quietly grant extra permissions |
| `max_work_usd`, `timeout_s` | The most one turn may cost, and how long it may take |

## About using your subscription

- This is for running agents on **your own work**. Anthropic's terms allow you to sign in to
  Claude Code with your own plan, and its usage limits "assume ordinary, individual usage of
  Claude Code and the Agent SDK."
- Workflows with many agents use up your plan's 5-hour and weekly limits faster, and they
  share those limits with the rest of your Claude use.
- Do not use a subscription to serve other people or to run a hosted product. Use an API key
  for that. See Anthropic's
  [legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance).
- If `ANTHROPIC_API_KEY` is set on your machine, Claude Code uses it and bills the API instead
  of your plan.
- Anthropic has changed these rules more than once, so check the current terms.

## Related work

If you already use LangGraph, [langchain-claude-cli](https://github.com/clriesco/langchain-claude-cli)
lets it run on a Claude subscription too, and LangGraph's workflow engine is more mature than
this one. agent-rooster is smaller, and it focuses on keeping each agent inside strict limits
on files and commands.

## Tests

```bash
uv run pytest            # free, with no calls to Claude
uv run pytest -m live    # calls the real Claude Code and uses a little of your plan
```

## License

MIT. See [LICENSE](LICENSE).
