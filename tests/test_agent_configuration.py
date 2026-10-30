"""Checks the parts of Agent that start no CLI process, so these cost nothing to run."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage
from pydantic import BaseModel

from rooster import Agent, AgentError, Command, Effort, Model, PermissionMode, Tool
from rooster.agent import _CLI_NAMES, _COMMAND_PATTERNS, _bash_rules, _cli_names


class Verdict(BaseModel):
    """Holds a reviewer's verdict, as a schema for the tests below."""

    verdict: str
    reason: str


@pytest.fixture
def make(tmp_path: Path):
    """Returns a function that builds an agent with a real profile and a log folder."""
    profile = tmp_path / "p.md"
    profile.write_text("x", encoding="utf-8")

    def build(**settings) -> Agent:
        settings.setdefault("cwd", tmp_path)
        settings.setdefault("log_dir", tmp_path / "logs")
        return Agent("a", profile, **settings)

    return build


def _options_for(agent: Agent) -> ClaudeAgentOptions:
    """Builds a first-turn set of options for inspection."""
    return agent._options(continuity=True)


def _result(**fields) -> ResultMessage:
    """Builds a finished turn's ResultMessage, with 'fields' replacing the defaults."""
    defaults = dict(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                    num_turns=1, session_id="s", total_cost_usd=0.01, result="")
    return ResultMessage(**{**defaults, **fields})


def test_missing_profile_raises(tmp_path: Path) -> None:
    """Proves a misconfigured agent fails at construction, before any subprocess runs."""
    with pytest.raises(AgentError, match="no profile at"):
        Agent("broken", tmp_path / "absent.md", cwd=tmp_path, log_dir=tmp_path)


def test_missing_cli_path_raises(make, tmp_path: Path) -> None:
    """Proves a cli_path that names nothing fails at construction."""
    with pytest.raises(AgentError, match="no claude executable"):
        make(cli_path=tmp_path / "claude.exe")


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


def test_full_asks_for_agent_and_no_retired_name(make) -> None:
    """Proves Tool.FULL asks the CLI for Agent, and for neither Task nor TodoWrite."""
    tools = _options_for(make(tools=Tool.FULL)).tools

    assert "Agent" in tools
    assert "Task" not in tools
    assert "TodoWrite" not in tools
    assert len(tools) == len(set(tools))


def test_enums_are_their_cli_strings() -> None:
    """Proves an enum member needs no conversion on the command line."""
    assert str(Model.SONNET) == "sonnet"
    assert str(PermissionMode.ACCEPT_EDITS) == "acceptEdits"
    assert str(Effort.HIGH) == "high"


def test_allowlist_becomes_cli_tool_names(make) -> None:
    """Proves a tool combination expands into the names the CLI expects."""
    options = _options_for(make(tools=Tool.INSPECT))

    assert set(options.tools) == {"Read", "Glob", "Grep"}
    assert set(options.allowed_tools) == {"Read", "Glob", "Grep"}


def test_empty_allowlist_denies_every_tool(make) -> None:
    """Proves Tool.NONE means no tools, rather than silently meaning all of them."""
    options = _options_for(make(tools=Tool.NONE))

    # None would leave the CLI's default set available, where an empty list means none at all
    assert options.tools == []
    assert options.allowed_tools == []


def test_bash_in_tools_allows_every_command(make) -> None:
    """Proves Tool.BASH approves Bash itself, and that unapproved calls are still denied."""
    options = _options_for(make(tools=Tool.BASH | Tool.READ))

    assert set(options.tools) == {"Bash", "Read"}
    assert set(options.allowed_tools) == {"Bash", "Read"}
    assert options.extra_args["permission-prompts"] == "none"


def test_every_command_has_patterns() -> None:
    """Proves each single Command flag maps to at least one command pattern."""
    singles = {member for member in Command.__members__.values() if member.value}
    assert singles == _COMMAND_PATTERNS.keys()
    assert all(_COMMAND_PATTERNS[member] for member in singles)


def test_a_command_makes_bash_available_for_it_alone(make) -> None:
    """Proves a Command makes Bash exist and approves only its own commands."""
    options = _options_for(make(tools=Tool.INSPECT, commands=Command.GIT_MV))

    assert "Bash" in options.tools
    assert "Bash(git mv *)" in options.allowed_tools
    assert "Bash" not in options.allowed_tools


def test_a_command_pattern_also_matches_its_bare_command() -> None:
    """Proves "pytest *" brings "pytest" with it, so a run with no arguments is allowed."""
    rules = _bash_rules(Command.PYTEST | Command.GIT_READ)

    assert "Bash(pytest *)" in rules
    assert "Bash(pytest)" in rules
    assert "Bash(uv run pytest)" in rules
    assert "Bash(git status)" in rules


def test_no_paths_leaves_edits_to_the_permission_mode(make) -> None:
    """Proves an agent without 'editable' approves edits anywhere, in accept-edits mode."""
    options = _options_for(make(tools=Tool.AUTHOR))

    assert {"Edit", "Write"} <= set(options.allowed_tools)
    assert options.permission_mode == "acceptEdits"
    assert options.disallowed_tools == []


def test_editable_approves_edits_only_by_path(make) -> None:
    """Proves 'editable' leaves the bare edit tools unapproved, adds a path rule for each
    pattern, and switches out of accept-edits mode, which would approve every edit."""
    options = _options_for(make(tools=Tool.AUTHOR, editable=("src/**", "tests/**")))

    assert {"Edit", "Write"} <= set(options.tools)
    assert "Edit" not in options.allowed_tools
    assert "Write" not in options.allowed_tools
    assert {"Edit(src/**)", "Edit(tests/**)", "Read"} <= set(options.allowed_tools)
    assert options.permission_mode == "dontAsk"


def test_frozen_becomes_a_deny_rule(make) -> None:
    """Proves each frozen pattern becomes a deny rule, with or without 'editable'."""
    options = _options_for(make(tools=Tool.AUTHOR, frozen=("**/*.feature",)))

    assert options.disallowed_tools == ["Edit(**/*.feature)"]


def test_editable_with_bypass_mode_raises(make) -> None:
    """Proves 'editable' fails at construction when bypass mode would skip its rules."""
    with pytest.raises(AgentError, match="BYPASS"):
        make(editable=("src/**",), permission_mode=PermissionMode.BYPASS)


def test_profile_is_appended_to_the_claude_code_prompt(make, tmp_path: Path) -> None:
    """Proves the profile adds to Claude Code's system prompt rather than replacing it."""
    options = _options_for(make())

    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert options.extra_args["append-system-prompt-file"] == str((tmp_path / "p.md").resolve())


def test_a_schema_class_becomes_its_json_schema(make) -> None:
    """Proves a pydantic class reaches the SDK as the JSON schema it describes."""
    options = _options_for(make(schema=Verdict))

    assert options.output_format == {"type": "json_schema", "schema": Verdict.model_json_schema()}


def test_a_reply_that_fits_the_schema_arrives_as_an_instance(make) -> None:
    """Proves structured output is turned into an instance of the schema class."""
    agent = make(schema=Verdict)

    result = agent._parse(_result(structured_output={"verdict": "ok", "reason": "fine"}), None)

    assert result["ok"]
    assert result["data"] == Verdict(verdict="ok", reason="fine")


def test_a_reply_in_plain_text_is_read_as_json(make) -> None:
    """Proves a reply with no structured output is read from its text."""
    agent = make(schema=Verdict)

    result = agent._parse(_result(result=json.dumps({"verdict": "ok", "reason": "fine"})), None)

    assert result["ok"]
    assert result["data"].verdict == "ok"


def test_a_reply_that_does_not_fit_the_schema_fails(make) -> None:
    """Proves a reply missing a required field is a failed turn that names the class."""
    agent = make(schema=Verdict)

    result = agent._parse(_result(structured_output={"verdict": "ok"}), None)

    assert result["ok"] is False
    assert result["data"] is None
    assert "Verdict" in result["failure"]


def test_a_new_agent_asks_for_no_particular_session(make) -> None:
    """Proves a first call carries no session, leaving the CLI to mint the id."""
    agent = make()

    options = _options_for(agent)

    assert agent.session_id is None
    assert options.session_id is None
    assert options.resume is None


def test_an_agent_holding_an_id_always_resumes(make) -> None:
    """Proves a saved id means rejoin, which is what makes a run survive a human gate."""
    saved = "28aff200-76ed-4718-a98a-0d67857acbb5"

    options = _options_for(make(session_id=saved))

    # a session_id on an existing session is rejected as "already in use", so it must never be set
    assert options.resume == saved
    assert options.session_id is None


def test_reset_forgets_the_session(make) -> None:
    """Proves reset abandons the conversation without discarding the result history."""
    agent = make(session_id="something")

    agent.reset()

    assert agent.session_id is None
    assert _options_for(agent).resume is None


def test_a_new_agent_has_an_empty_transcript(make) -> None:
    """Proves an agent that has never run returns an empty transcript."""
    assert make().transcript() == []


def test_run_fresh_never_touches_the_session(make) -> None:
    """Proves a throwaway turn leaves the agent's own conversation alone."""
    saved = "28aff200-76ed-4718-a98a-0d67857acbb5"
    agent = make(session_id=saved)

    options = agent._options(continuity=False)

    assert options.resume is None
    assert agent.session_id == saved


def test_a_fork_resumes_a_copy_of_the_session(make) -> None:
    """Proves a forked turn resumes this agent's session and asks the CLI to copy it."""
    saved = "28aff200-76ed-4718-a98a-0d67857acbb5"

    options = make(session_id=saved)._options(continuity=False, fork=True)

    assert options.resume == saved
    assert options.fork_session is True


def test_safe_mode_controls_the_flag(make) -> None:
    """Proves safe_mode is what puts --safe-mode on the command line."""
    assert "safe-mode" in _options_for(make()).extra_args
    assert "safe-mode" not in _options_for(make(safe_mode=False)).extra_args


def test_effort_reaches_the_options(make) -> None:
    """Proves an effort level is passed to the SDK as its CLI string."""
    assert _options_for(make(effort=Effort.HIGH)).effort == "high"


def test_each_message_is_logged_as_it_arrives(make, tmp_path: Path, monkeypatch) -> None:
    """Proves the log file holds each message before the turn ends, so it can be watched."""
    seen_in_log: list[str] = []
    log_dir = tmp_path / "logs"

    async def fake_query(prompt, options):
        yield _result(result="first")
        # read the log while the turn is still running
        seen_in_log.append(next(log_dir.iterdir()).read_text(encoding="utf-8"))
        yield _result(result="second")

    monkeypatch.setattr("rooster.agent.query", fake_query)
    result = make().run_fresh("hi")

    assert '"result": "first"' in seen_in_log[0]
    log = result["log_path"].read_text(encoding="utf-8")
    assert log.index("--- options ---") < log.index("--- messages ---") < log.index("--- stderr ---")
    assert '"result": "second"' in log


def test_an_unknown_session_is_expired_without_running(make, tmp_path: Path) -> None:
    """Proves a session the CLI does not hold is reported as expired before anything spawns."""
    gone = str(uuid.uuid4())
    # a cli_path that cannot run proves nothing spawned, since spawning it would raise
    fake_cli = tmp_path / "claude.exe"
    fake_cli.write_text("", encoding="utf-8")
    agent = make(session_id=gone, cli_path=fake_cli)

    result = agent.run("Anything.")

    assert result["ok"] is False
    assert result["session_expired"] is True
    assert result["cost_usd"] == 0.0
    assert agent.session_id == gone


def test_total_tokens_counts_cache_traffic(make) -> None:
    """Proves the running total includes cache reads, which a resumed turn arrives as."""
    agent = make()

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
