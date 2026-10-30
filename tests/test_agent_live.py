"""Checks Agent against the real claude CLI; every test here spends tokens."""

from __future__ import annotations

from pathlib import Path

import pytest

from rooster import Agent, Model, Tool

pytestmark = pytest.mark.live


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


