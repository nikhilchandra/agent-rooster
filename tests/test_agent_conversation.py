"""Checks Agent against the real claude CLI; every test here spends tokens."""

from __future__ import annotations

import asyncio
import json
import time
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from rooster import Agent, Command, Model, Tool
from rooster.agent import _ALIASES, _CLI_NAMES

pytestmark = pytest.mark.live

# tools that only start once a plugin or an MCP server is configured
NEEDS_SETUP = {"LSP", "ListMcpResourcesTool", "ReadMcpResourceTool"}


class Verdict(BaseModel):
    """Holds a reviewer's verdict on a patch."""

    verdict: Literal["approved", "changes_requested"]
    reason: str


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Gives each test its own directory for the agent to run in."""
    return tmp_path


@pytest.fixture
def logs(workspace: Path) -> Path:
    """Gives each test's agents a log folder inside its workspace."""
    return workspace / "logs"


@pytest.fixture
def plain_profile(workspace: Path) -> Path:
    """Writes a profile that adds no persona, for tests about the mechanism itself."""
    path = workspace / "plain.md"
    path.write_text("Answer exactly what is asked, with no preamble.", encoding="utf-8")
    return path


def test_envelope_parses_and_context_is_clean(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves the JSON envelope parses and that safe mode keeps the reply bare."""
    agent = Agent("plain", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE)

    result = agent.run("Reply with exactly: ok")

    assert result["ok"], result["failure"]
    # a reply of exactly "ok" shows no CLAUDE.md footer or other personal instructions leaked in
    assert result["text"].strip() == "ok"
    assert result["session_id"]
    assert result["log_path"] is not None and result["log_path"].is_file()
    assert agent.total_tokens > 0


def test_profile_reaches_the_model(workspace: Path, logs: Path) -> None:
    """Proves an appended profile survives safe mode and shapes the reply."""
    profile = workspace / "reviewer.md"
    profile.write_text(
        "You are REVIEWER-7. Begin every reply with the literal token [REVIEWER-7].",
        encoding="utf-8",
    )
    agent = Agent("reviewer", profile, cwd=workspace, log_dir=logs, tools=Tool.NONE)

    result = agent.run("State your designation in one short line.")

    assert result["ok"], result["failure"]
    assert "[REVIEWER-7]" in result["text"]


def test_schema_returns_an_instance(workspace: Path, logs: Path, plain_profile: Path) -> None:
    """Proves a schema-constrained reply arrives as an instance of the schema class."""
    agent = Agent("gate", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE,
                  schema=Verdict)

    result = agent.run("A patch deletes a test so the suite passes. Give your verdict.")

    assert result["ok"], result["failure"]
    assert isinstance(result["data"], Verdict)
    assert result["data"].verdict in {"approved", "changes_requested"}


def test_session_continues_across_calls(workspace: Path, logs: Path, plain_profile: Path) -> None:
    """Proves a second run rejoins the first, so the agent remembers its own turns."""
    agent = Agent("memory", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE)

    first = agent.run("Remember the number 41. Reply with exactly: stored")
    assert first["ok"], first["failure"]

    second = agent.run("What number did I ask you to remember? Reply with just the number.")
    assert second["ok"], second["failure"]
    # the agent could only know this from the earlier turn in the same session
    assert "41" in second["text"]
    assert second["session_id"] == first["session_id"]

    # the cache is reported rather than asserted: a two-turn exchange can fall under the
    # minimum size the API will cache, so a zero here is not a failure of continuity
    print(f"\ncache read on turn 2: {second['cache_read_tokens']} tokens")
    print(f"total across both turns: {agent.total_tokens} tokens, ${agent.total_cost_usd:.4f}")


def test_two_agents_run_at_the_same_time(workspace: Path, logs: Path, plain_profile: Path) -> None:
    """Proves arun_fresh lets two agents work at once, so the second starts before the first
    finishes."""
    first = Agent("first", plain_profile, cwd=workspace, log_dir=logs, model=Model.HAIKU,
                  tools=Tool.NONE)
    second = Agent("second", plain_profile, cwd=workspace, log_dir=logs, model=Model.HAIKU,
                   tools=Tool.NONE)

    async def timed(agent: Agent) -> tuple[float, float, dict]:
        """Returns when the agent's turn started and ended, in seconds, and its result."""
        start = time.monotonic()
        result = await agent.arun_fresh("Reply with exactly: ok")
        return start, time.monotonic(), result

    async def both() -> list:
        """Runs the two agents together."""
        return await asyncio.gather(timed(first), timed(second))

    (first_start, first_end, first_result), (second_start, _, second_result) = asyncio.run(both())

    assert first_result["ok"], first_result["failure"]
    assert second_result["ok"], second_result["failure"]
    # a turn that blocked the event loop would let the second start only after the first ended
    assert second_start < first_end
    print(f"\nboth finished in {first_end - first_start:.1f}s of the first agent's time")


def test_a_second_agent_object_resumes_the_first(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves a rebuilt agent rejoins the conversation, which is what surviving a gate needs."""
    first = Agent("memory", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE)
    opening = first.run("Remember the number 63. Reply with exactly: stored")
    assert opening["ok"], opening["failure"]
    saved_id = first.session_id
    assert saved_id

    # a separate object, exactly as a later process would build it after a gate
    second = Agent("memory", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE,
                   session_id=saved_id)
    result = second.run("What number did I ask you to remember? Reply with just the number.")

    assert result["ok"], result["failure"]
    assert "63" in result["text"]
    assert second.session_id == saved_id


def test_an_expired_session_is_reported_and_costs_nothing(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves a vanished session is named as such rather than silently replaced."""
    gone = "00000000-0000-4000-8000-000000000000"
    agent = Agent("ghost", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE,
                  session_id=gone)

    result = agent.run("Anything.")

    assert result["ok"] is False
    assert result["session_expired"] is True
    assert result["cost_usd"] == 0.0
    # the agent must not quietly start over, since the caller's prompt assumed a memory
    assert agent.session_id == gone


def test_non_ascii_text_survives_the_round_trip(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves accented letters and dashes arrive intact rather than mangled by a Windows
    codepage."""
    agent = Agent("unicode", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE)

    result = agent.run("Reply with exactly this word and nothing else: naïve—café")

    assert result["ok"], result["failure"]
    assert "naïve—café" in result["text"]


def test_a_prompt_longer_than_the_windows_command_line_works(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves a prompt past the 32,767-character Windows command-line limit still arrives."""
    agent = Agent("long", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE)
    filler = "filler " * 6000
    assert len(filler) > 40_000

    result = agent.run(f"Ignore the filler text below. Reply with exactly: ok\n\n{filler}")

    assert result["ok"], result["failure"]
    assert result["text"].strip() == "ok"


def test_the_cli_provides_every_named_tool(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves the CLI starts with every tool that _CLI_NAMES names, apart from the ones that
    need a plugin or an MCP server, so none is silently dropped as a name the CLI no longer
    knows."""
    agent = Agent("every", plain_profile, cwd=workspace, log_dir=logs, model=Model.HAIKU,
                  tools=reduce(or_, _CLI_NAMES))

    result = agent.run("Reply with exactly: ok")

    assert result["ok"], result["failure"]
    # read the logged messages, one JSON object per line between their two section markers
    log = result["log_path"].read_text(encoding="utf-8")
    section = log.split("--- messages ---\n", 1)[1].split("\n--- stderr ---", 1)[0]
    messages = [json.loads(line) for line in section.splitlines() if line.strip()]

    # the init message lists the tools the CLI actually started with
    init = next(
        m for m in messages if m["type"] == "SystemMessage" and m.get("subtype") == "init"
    )
    started = {_ALIASES.get(name, name) for name in init["data"]["tools"]}
    missing = set(_CLI_NAMES.values()) - NEEDS_SETUP - started
    assert not missing, f"the CLI did not start with {sorted(missing)}"


def test_a_timeout_is_reported_as_a_failure(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves a turn that overruns timeout_s comes back as a failed result, not an exception."""
    agent = Agent("slow", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.NONE,
                  timeout_s=2)

    result = agent.run("Write the numbers from 1 to 2000, one per line.")

    assert result["ok"] is False
    assert "timed out" in result["failure"]


def test_editable_and_frozen_limit_where_the_agent_writes(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves an agent can write inside 'editable', and is refused outside it and on a
    'frozen' path inside it."""
    (workspace / "src").mkdir()
    (workspace / "other").mkdir()
    agent = Agent("guarded", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.AUTHOR,
                  editable=("src/**",), frozen=("**/*.feature",))

    result = agent.run(
        "Use the Write tool three times, once for each of these files, each containing the "
        "word hi: src/ok.txt, other/no.txt, src/x.feature. Attempt all three even if one is "
        "refused. Then list which ones were refused."
    )

    assert result["ok"], result["failure"]
    assert (workspace / "src" / "ok.txt").is_file()
    assert not (workspace / "other" / "no.txt").exists()
    assert not (workspace / "src" / "x.feature").exists()
    assert len(result["denials"]) >= 2


def test_a_command_runs_and_any_other_is_refused(
    workspace: Path, logs: Path, plain_profile: Path
) -> None:
    """Proves Command.PYTEST lets the agent run pytest and refuses a command it does not name."""
    (workspace / "test_one.py").write_text("def test_one():\n    assert True\n", encoding="utf-8")
    agent = Agent("tester", plain_profile, cwd=workspace, log_dir=logs, tools=Tool.INSPECT,
                  commands=Command.PYTEST)

    # not `echo`, which the CLI treats as read-only and approves without any rule
    result = agent.run(
        'Use the Bash tool to run exactly: python -m pytest -q test_one.py  then run exactly: '
        'python -c "print(6*7)"  Report each output. If a call is refused, say so plainly and '
        "do not guess its output."
    )

    assert result["ok"], result["failure"]
    assert "1 passed" in result["text"]
    assert any(d["tool_name"] == "Bash" for d in result["denials"])
    assert "42" not in result["text"]
