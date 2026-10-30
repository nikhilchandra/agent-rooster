"""Checks Agent against the real claude CLI; every test here spends tokens."""

from __future__ import annotations

import json
from functools import reduce
from operator import or_
from pathlib import Path

import pytest

from rooster import Agent, Model, Tool
from rooster.agent import _ALIASES, _CLI_NAMES

pytestmark = pytest.mark.live

# tools that only start once a plugin or an MCP server is configured
NEEDS_SETUP = {"LSP", "ListMcpResourcesTool", "ReadMcpResourceTool"}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Gives each test its own directory for the agent to run in."""
    return tmp_path


@pytest.fixture
def plain_profile(workspace: Path) -> Path:
    """Writes a profile that adds no persona, for tests about the mechanism itself."""
    path = workspace / "plain.md"
    path.write_text("Answer exactly what is asked, with no preamble.", encoding="utf-8")
    return path


def test_envelope_parses_and_context_is_clean(workspace: Path, plain_profile: Path) -> None:
    """Proves the JSON envelope parses and that clean_context keeps the reply bare."""
    agent = Agent(
        "plain",
        plain_profile,
        cwd=workspace,
        allowed_tools=Tool.NONE,
        log_dir=workspace / "logs",
    )

    result = agent.run("Reply with exactly: ok")

    assert result["ok"], result["failure"]
    # a reply of exactly "ok" shows no CLAUDE.md footer or other personal instructions leaked in
    assert result["text"].strip() == "ok"
    assert result["session_id"]
    assert result["log_path"] is not None and result["log_path"].is_file()
    assert agent.total_tokens > 0


def test_profile_reaches_the_model(workspace: Path) -> None:
    """Proves an appended profile survives safe mode and shapes the reply."""
    profile = workspace / "reviewer.md"
    profile.write_text(
        "You are REVIEWER-7. Begin every reply with the literal token [REVIEWER-7].",
        encoding="utf-8",
    )
    agent = Agent("reviewer", profile, cwd=workspace, allowed_tools=Tool.NONE)

    result = agent.run("State your designation in one short line.")

    assert result["ok"], result["failure"]
    assert "[REVIEWER-7]" in result["text"]


def test_schema_returns_parsed_data(workspace: Path, plain_profile: Path) -> None:
    """Proves a schema-constrained reply arrives as a dict, needing no parser."""
    verdict = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["approved", "changes_requested"]},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    }
    agent = Agent(
        "gate",
        plain_profile,
        cwd=workspace,
        allowed_tools=Tool.NONE,
        schema=verdict,
    )

    result = agent.run("A patch deletes a test so the suite passes. Give your verdict.")

    assert result["ok"], result["failure"]
    assert isinstance(result["data"], dict)
    assert result["data"]["verdict"] in {"approved", "changes_requested"}


def test_session_continues_across_calls(workspace: Path, plain_profile: Path) -> None:
    """Proves a second run rejoins the first, so the agent remembers its own turns."""
    agent = Agent("memory", plain_profile, cwd=workspace, allowed_tools=Tool.NONE)

    first = agent.run("Remember the number 41. Reply with exactly: stored")
    assert first["ok"], first["failure"]

    second = agent.run("What number did I ask you to remember? Reply with just the number.")
    assert second["ok"], second["failure"]
    # the agent could only know this from the earlier turn in the same session
    assert "41" in second["text"]
    assert second["session_id"] == first["session_id"]
    assert len(agent.results) == 2

    # the cache is reported rather than asserted: a two-turn exchange can fall under the
    # minimum size the API will cache, so a zero here is not a failure of continuity
    print(f"\ncache read on turn 2: {second['cache_read_tokens']} tokens")
    print(f"total across both turns: {agent.total_tokens} tokens, ${agent.total_cost_usd:.4f}")


def test_a_forked_question_remembers_without_moving_the_session(
    workspace: Path, plain_profile: Path
) -> None:
    """Proves a forked turn sees the agent's memory and leaves its own session where it was."""
    agent = Agent("memory", plain_profile, cwd=workspace, allowed_tools=Tool.NONE)
    opening = agent.run("Remember the number 77. Reply with exactly: stored")
    assert opening["ok"], opening["failure"]
    before = agent.session_id

    result = agent.run_forked("What number did I ask you to remember? Reply with just the number.")

    assert result["ok"], result["failure"]
    assert "77" in result["text"]
    assert result["session_id"] != before
    assert agent.session_id == before


def test_a_fork_with_nothing_to_fork_from_fails(workspace: Path, plain_profile: Path) -> None:
    """Proves forking an agent that has never run fails before any CLI call."""
    agent = Agent("new", plain_profile, cwd=workspace, allowed_tools=Tool.NONE)

    result = agent.run_forked("Anything.")

    assert result["ok"] is False
    assert agent.session_id is None



def test_a_second_agent_object_resumes_the_first(workspace: Path, plain_profile: Path) -> None:
    """Proves a rebuilt agent rejoins the conversation, which is what surviving a gate needs."""
    first = Agent("memory", plain_profile, cwd=workspace, allowed_tools=Tool.NONE)
    opening = first.run("Remember the number 63. Reply with exactly: stored")
    assert opening["ok"], opening["failure"]
    saved_id = first.session_id
    assert saved_id

    # a separate object, exactly as a later `bdd continue` process would build it
    second = Agent("memory", plain_profile, cwd=workspace, allowed_tools=Tool.NONE,
                   session_id=saved_id)
    result = second.run("What number did I ask you to remember? Reply with just the number.")

    assert result["ok"], result["failure"]
    assert "63" in result["text"]
    assert second.session_id == saved_id


def test_an_expired_session_is_reported_and_costs_nothing(workspace: Path, plain_profile: Path) -> None:
    """Proves a vanished session is named as such rather than silently replaced."""
    gone = "00000000-0000-4000-8000-000000000000"
    agent = Agent("ghost", plain_profile, cwd=workspace, allowed_tools=Tool.NONE,
                  session_id=gone)

    result = agent.run("Anything.")

    assert result["ok"] is False
    assert result["session_expired"] is True
    assert result["cost_usd"] == 0.0
    # the agent must not quietly start over, since the caller's prompt assumed a memory
    assert agent.session_id == gone


def test_non_ascii_text_survives_the_round_trip(workspace: Path, plain_profile: Path) -> None:
    """Proves accented letters and dashes arrive intact rather than mangled by a Windows
    codepage."""
    agent = Agent("unicode", plain_profile, cwd=workspace, allowed_tools=Tool.NONE)

    result = agent.run("Reply with exactly this word and nothing else: naïve—café")

    assert result["ok"], result["failure"]
    assert "naïve—café" in result["text"]


def test_a_prompt_longer_than_the_windows_command_line_works(
    workspace: Path, plain_profile: Path
) -> None:
    """Proves a prompt past the 32,767-character Windows command-line limit still arrives."""
    agent = Agent("long", plain_profile, cwd=workspace, allowed_tools=Tool.NONE)
    filler = "filler " * 6000
    assert len(filler) > 40_000

    result = agent.run(f"Ignore the filler text below. Reply with exactly: ok\n\n{filler}")

    assert result["ok"], result["failure"]
    assert result["text"].strip() == "ok"


def test_the_cli_provides_every_named_tool(workspace: Path, plain_profile: Path) -> None:
    """Proves the CLI starts with every tool that _CLI_NAMES names, apart from the ones that
    need a plugin or an MCP server, so none is silently dropped as a name the CLI no longer
    knows."""
    agent = Agent(
        "every",
        plain_profile,
        cwd=workspace,
        model=Model.HAIKU,
        allowed_tools=reduce(or_, _CLI_NAMES),
        log_dir=workspace / "logs",
    )

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


def test_a_timeout_is_reported_as_a_failure(workspace: Path, plain_profile: Path) -> None:
    """Proves a turn that overruns timeout_s comes back as a failed result, not an exception."""
    agent = Agent("slow", plain_profile, cwd=workspace, allowed_tools=Tool.NONE, timeout_s=2)

    result = agent.run("Write the numbers from 1 to 2000, one per line.")

    assert result["ok"] is False
    assert "timed out" in result["failure"]
