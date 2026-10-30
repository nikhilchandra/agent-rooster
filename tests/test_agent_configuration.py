"""Checks the parts of Agent that need no subprocess, so these cost nothing to run."""

from __future__ import annotations

from pathlib import Path

import pytest

from rooster import Agent, AgentError, Model, PermissionMode, Tool


def test_missing_profile_raises(tmp_path: Path) -> None:
    """Proves a misconfigured agent fails at construction, before any subprocess runs."""
    with pytest.raises(AgentError, match="no profile at"):
        Agent("broken", tmp_path / "absent.md", cwd=tmp_path)


def test_tool_flags_compose() -> None:
    """Proves the tool bitmask supports composites, membership and subtraction."""
    assert Tool.READ in Tool.INSPECT
    assert Tool.EDIT not in Tool.INSPECT
    assert Tool.EDIT in Tool.AUTHOR
    assert Tool.BASH not in (Tool.FULL & ~Tool.BASH)


def test_enums_are_their_cli_strings() -> None:
    """Proves an enum member needs no conversion on the command line."""
    assert str(Model.SONNET) == "sonnet"
    assert str(PermissionMode.ACCEPT_EDITS) == "acceptEdits"


def _argv_for(agent: Agent) -> list[str]:
    """Builds a first-turn command line for inspection."""
    return agent._argv(continuity=True)


def test_allowlist_becomes_cli_tool_names(tmp_path: Path) -> None:
    """Proves a tool combination expands into the names the CLI expects."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path, allowed_tools=Tool.INSPECT)

    argv = _argv_for(agent)

    assert "--allowed-tools" in argv
    names = argv[argv.index("--allowed-tools") + 1 : argv.index("--allowed-tools") + 4]
    assert set(names) == {"Read", "Glob", "Grep"}


def test_empty_allowlist_denies_every_tool(tmp_path: Path) -> None:
    """Proves Tool.NONE means no tools, rather than silently meaning all of them."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path, allowed_tools=Tool.NONE)

    argv = _argv_for(agent)

    # omitting --allowed-tools would leave every tool available, so the flags must invert
    assert "--allowed-tools" not in argv
    assert "--disallowed-tools" in argv
    assert "Bash" in argv and "Edit" in argv


def test_a_new_agent_asks_for_no_particular_session(tmp_path: Path) -> None:
    """Proves a first call carries no session flag, leaving the CLI to mint the id."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path)

    argv = _argv_for(agent)

    assert agent.session_id is None
    assert "--session-id" not in argv
    assert "--resume" not in argv


def test_an_agent_holding_an_id_always_resumes(tmp_path: Path) -> None:
    """Proves a saved id means rejoin, which is what makes a run survive a human gate."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    saved = "28aff200-76ed-4718-a98a-0d67857acbb5"

    agent = Agent("a", profile, cwd=tmp_path, session_id=saved)

    argv = _argv_for(agent)
    # --session-id on an existing session is rejected as "already in use", so it must never appear
    assert "--resume" in argv and saved in argv
    assert "--session-id" not in argv


def test_reset_forgets_the_session(tmp_path: Path) -> None:
    """Proves reset abandons the conversation without discarding the result history."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path, session_id="something")

    agent.reset()

    assert agent.session_id is None
    assert "--resume" not in _argv_for(agent)


def test_run_fresh_never_touches_the_session(tmp_path: Path) -> None:
    """Proves a throwaway turn leaves the agent's own conversation alone."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    saved = "28aff200-76ed-4718-a98a-0d67857acbb5"
    agent = Agent("a", profile, cwd=tmp_path, session_id=saved)

    argv = agent._argv(continuity=False)

    assert "--resume" not in argv
    assert agent.session_id == saved


def test_clean_context_controls_safe_mode(tmp_path: Path) -> None:
    """Proves clean_context is what puts --safe-mode on the command line."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")

    assert "--safe-mode" in _argv_for(Agent("a", profile, cwd=tmp_path))
    assert "--safe-mode" not in _argv_for(
        Agent("a", profile, cwd=tmp_path, clean_context=False)
    )


def test_total_tokens_counts_cache_traffic(tmp_path: Path) -> None:
    """Proves the running total includes cache reads, which a resumed turn arrives as."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path)

    # a resumed turn's context is nearly all cache reads; counting only input plus
    # output would report 11 tokens for an exchange that actually moved almost 6,000
    agent._record(
        {
            "cost_usd": 0.01,
            "input_tokens": 4,
            "output_tokens": 7,
            "cache_read_tokens": 5910,
            "cache_write_tokens": 0,
            "total_tokens": 4 + 7 + 5910,
        }
    )

    assert agent.total_tokens == 5921
