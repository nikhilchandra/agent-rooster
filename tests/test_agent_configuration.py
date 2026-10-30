"""Checks the parts of Agent that start no CLI process, so these cost nothing to run."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from rooster import Agent, AgentError, Effort, Model, PermissionMode, Tool
from rooster.agent import _CLI_NAMES, _cli_names


def test_missing_profile_raises(tmp_path: Path) -> None:
    """Proves a misconfigured agent fails at construction, before any subprocess runs."""
    with pytest.raises(AgentError, match="no profile at"):
        Agent("broken", tmp_path / "absent.md", cwd=tmp_path)


def test_missing_cli_path_raises(tmp_path: Path) -> None:
    """Proves a cli_path that names nothing fails at construction."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    with pytest.raises(AgentError, match="no claude executable"):
        Agent("broken", profile, cwd=tmp_path, cli_path=tmp_path / "claude.exe")


def test_tool_flags_compose() -> None:
    """Proves the tool bitmask supports composites, membership and subtraction."""
    assert Tool.READ in Tool.INSPECT
    assert Tool.EDIT not in Tool.INSPECT
    assert Tool.EDIT in Tool.AUTHOR
    assert Tool.BASH not in (Tool.FULL & ~Tool.BASH)


def test_every_tool_has_its_own_cli_name() -> None:
    """Proves each single tool has a CLI name, and that no two of them share one."""
    # the single-bit members, skipping NONE, the composites and any alias of an earlier member
    singles = {
        member
        for member in Tool.__members__.values()
        if member.value and member.value & (member.value - 1) == 0
    }
    assert singles == _CLI_NAMES.keys()
    assert len(set(_CLI_NAMES.values())) == len(_CLI_NAMES)


def test_full_holds_only_the_core_tools() -> None:
    """Proves the tools that only some setups need stay out of Tool.FULL."""
    assert set(_cli_names(Tool.FULL)) == {
        "Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent", "WebFetch", "WebSearch",
    }


def test_tasks_expands_to_the_four_task_tools() -> None:
    """Proves Tool.TASKS asks for the four tools that track a task list."""
    assert set(_cli_names(Tool.TASKS)) == {"TaskCreate", "TaskGet", "TaskList", "TaskUpdate"}


def test_full_asks_for_agent_and_no_retired_name(tmp_path: Path) -> None:
    """Proves Tool.FULL asks the CLI for Agent, and for neither Task nor TodoWrite."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")

    tools = Agent("a", profile, cwd=tmp_path, allowed_tools=Tool.FULL)._options(continuity=True).tools

    assert "Agent" in tools
    assert "Task" not in tools
    assert "TodoWrite" not in tools
    assert len(tools) == len(set(tools))


def test_enums_are_their_cli_strings() -> None:
    """Proves an enum member needs no conversion on the command line."""
    assert str(Model.SONNET) == "sonnet"
    assert str(PermissionMode.ACCEPT_EDITS) == "acceptEdits"
    assert str(Effort.HIGH) == "high"


def _options_for(agent: Agent) -> ClaudeAgentOptions:
    """Builds a first-turn set of options for inspection."""
    return agent._options(continuity=True)


def test_allowlist_becomes_cli_tool_names(tmp_path: Path) -> None:
    """Proves a tool combination expands into the names the CLI expects."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path, allowed_tools=Tool.INSPECT)

    options = _options_for(agent)

    assert set(options.tools) == {"Read", "Glob", "Grep"}
    assert set(options.allowed_tools) == {"Read", "Glob", "Grep"}


def test_empty_allowlist_denies_every_tool(tmp_path: Path) -> None:
    """Proves Tool.NONE means no tools, rather than silently meaning all of them."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path, allowed_tools=Tool.NONE)

    options = _options_for(agent)

    # None would leave the CLI's default set available, where an empty list means none at all
    assert options.tools == []
    assert options.allowed_tools == []


def test_an_mcp_rule_adds_no_builtin_tool(tmp_path: Path) -> None:
    """Proves allowing an MCP tool leaves the built-in tools as the flags set them."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path, allowed_tools=Tool.NONE,
                  extra_allowed_tools=("mcp__x__y",))

    options = _options_for(agent)

    assert options.tools == []
    assert options.allowed_tools == ["mcp__x__y"]


def test_a_pattern_rule_makes_its_tool_available(tmp_path: Path) -> None:
    """Proves a rule like "Bash(git *)" makes Bash exist, or the rule could never apply."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path, allowed_tools=Tool.INSPECT,
                  extra_allowed_tools=("Bash(git *)",))

    options = _options_for(agent)

    assert "Bash" in options.tools
    assert "Bash(git *)" in options.allowed_tools
    assert "Bash" not in options.allowed_tools


def test_profile_is_appended_to_the_claude_code_prompt(tmp_path: Path) -> None:
    """Proves the profile adds to Claude Code's system prompt rather than replacing it."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")

    options = _options_for(Agent("a", profile, cwd=tmp_path))

    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert options.extra_args["append-system-prompt-file"] == str(profile.resolve())


def test_a_new_agent_asks_for_no_particular_session(tmp_path: Path) -> None:
    """Proves a first call carries no session, leaving the CLI to mint the id."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path)

    options = _options_for(agent)

    assert agent.session_id is None
    assert options.session_id is None
    assert options.resume is None


def test_an_agent_holding_an_id_always_resumes(tmp_path: Path) -> None:
    """Proves a saved id means rejoin, which is what makes a run survive a human gate."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    saved = "28aff200-76ed-4718-a98a-0d67857acbb5"

    agent = Agent("a", profile, cwd=tmp_path, session_id=saved)

    options = _options_for(agent)
    # a session_id on an existing session is rejected as "already in use", so it must never be set
    assert options.resume == saved
    assert options.session_id is None


def test_reset_forgets_the_session(tmp_path: Path) -> None:
    """Proves reset abandons the conversation without discarding the result history."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path, session_id="something")

    agent.reset()

    assert agent.session_id is None
    assert _options_for(agent).resume is None


def test_a_new_agent_has_an_empty_transcript(tmp_path: Path) -> None:
    """Proves an agent that has never run returns an empty transcript."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    agent = Agent("a", profile, cwd=tmp_path)

    assert agent.transcript() == []


def test_run_fresh_never_touches_the_session(tmp_path: Path) -> None:
    """Proves a throwaway turn leaves the agent's own conversation alone."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    saved = "28aff200-76ed-4718-a98a-0d67857acbb5"
    agent = Agent("a", profile, cwd=tmp_path, session_id=saved)

    options = agent._options(continuity=False)

    assert options.resume is None
    assert agent.session_id == saved


def test_a_fork_resumes_a_copy_of_the_session(tmp_path: Path) -> None:
    """Proves a forked turn resumes this agent's session and asks the CLI to copy it."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    saved = "28aff200-76ed-4718-a98a-0d67857acbb5"
    agent = Agent("a", profile, cwd=tmp_path, session_id=saved)

    options = agent._options(continuity=False, fork=True)

    assert options.resume == saved
    assert options.fork_session is True


def test_clean_context_controls_safe_mode(tmp_path: Path) -> None:
    """Proves clean_context is what puts --safe-mode on the command line."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")

    assert "safe-mode" in _options_for(Agent("a", profile, cwd=tmp_path)).extra_args
    assert "safe-mode" not in _options_for(
        Agent("a", profile, cwd=tmp_path, clean_context=False)
    ).extra_args


def test_effort_reaches_the_options(tmp_path: Path) -> None:
    """Proves an effort level is passed to the SDK as its CLI string."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")

    options = _options_for(Agent("a", profile, cwd=tmp_path, effort=Effort.HIGH))

    assert options.effort == "high"


def test_max_turns_reaches_the_options(tmp_path: Path) -> None:
    """Proves a turn limit is passed to the SDK unchanged."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")

    options = _options_for(Agent("a", profile, cwd=tmp_path, max_turns=5))

    assert options.max_turns == 5


def test_fallback_model_reaches_the_options(tmp_path: Path) -> None:
    """Proves a fallback model is passed to the SDK unchanged."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")

    options = _options_for(Agent("a", profile, cwd=tmp_path, fallback_model="claude-haiku-4-5"))

    assert options.fallback_model == "claude-haiku-4-5"


def test_an_unknown_session_is_expired_without_running(tmp_path: Path) -> None:
    """Proves a session the CLI does not hold is reported as expired before anything spawns."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")
    gone = str(uuid.uuid4())
    # a cli_path that cannot run proves nothing spawned, since spawning it would raise
    fake_cli = tmp_path / "claude.exe"
    fake_cli.write_text("", encoding="utf-8")
    agent = Agent("ghost", profile, cwd=tmp_path, session_id=gone, cli_path=fake_cli)

    result = agent.run("Anything.")

    assert result["ok"] is False
    assert result["session_expired"] is True
    assert result["cost_usd"] == 0.0
    assert agent.session_id == gone


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
