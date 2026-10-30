"""Runs one Claude Code agent through the Agent SDK and returns its answer as data."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Callable
from contextlib import aclosing
from datetime import datetime, timezone
from enum import Flag, StrEnum, auto
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLINotFoundError,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SessionMessage,
    ToolPermissionContext,
    get_session_info,
    get_session_messages,
    query,
)


class AgentError(RuntimeError):
    """Signals a problem that retrying will not fix."""


class Tool(Flag):
    """Names the tools an agent may use, combined with the | operator."""

    NONE = 0
    READ = auto()
    WRITE = auto()
    EDIT = auto()
    BASH = auto()
    GLOB = auto()
    GREP = auto()
    AGENT = auto()
    TASK = AGENT # the name from before CLI 2.1.63
    WEB_FETCH = auto()
    WEB_SEARCH = auto()

    # these suit only some setups, so no composite below includes them
    NOTEBOOK_EDIT = auto()
    POWERSHELL = auto()
    TASK_CREATE = auto()
    TASK_GET = auto()
    TASK_LIST = auto()
    TASK_UPDATE = auto()
    TASK_STOP = auto()
    SKILL = auto()
    LIST_MCP_RESOURCES = auto()
    READ_MCP_RESOURCE = auto()
    LSP = auto()
    ENTER_WORKTREE = auto()
    EXIT_WORKTREE = auto()
    MONITOR = auto()

    # composites, so a caller writes the intent rather than assembling it every time
    INSPECT = READ | GLOB | GREP
    AUTHOR = READ | GLOB | GREP | WRITE | EDIT
    FULL = READ | GLOB | GREP | WRITE | EDIT | BASH | AGENT | WEB_FETCH | WEB_SEARCH
    TASKS = TASK_CREATE | TASK_GET | TASK_LIST | TASK_UPDATE


class PermissionMode(StrEnum):
    """Names the permission modes the CLI accepts; exactly one applies at a time."""

    ACCEPT_EDITS = "acceptEdits"
    AUTO = "auto"
    BYPASS = "bypassPermissions"
    MANUAL = "manual"
    DONT_ASK = "dontAsk"
    PLAN = "plan"


class Model(StrEnum):
    """Names a model family; a full model ID may be passed as a plain string instead."""

    SONNET = "sonnet"
    OPUS = "opus"
    HAIKU = "haiku"
    FABLE = "fable"


class Effort(StrEnum):
    """Names the effort levels the CLI accepts, from least to most thorough."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


# Maps each single-bit Tool to the name the CLI knows it by. This is a subset of the tools
# listed at https://code.claude.com/docs/en/tools-reference, and any other tool can still be
# allowed by name through extra_allowed_tools.
_CLI_NAMES: dict[Tool, str] = {
    Tool.READ: "Read",
    Tool.WRITE: "Write",
    Tool.EDIT: "Edit",
    Tool.BASH: "Bash",
    Tool.GLOB: "Glob",
    Tool.GREP: "Grep",
    Tool.AGENT: "Agent",
    Tool.WEB_FETCH: "WebFetch",
    Tool.WEB_SEARCH: "WebSearch",
    Tool.NOTEBOOK_EDIT: "NotebookEdit",
    Tool.POWERSHELL: "PowerShell",
    Tool.TASK_CREATE: "TaskCreate",
    Tool.TASK_GET: "TaskGet",
    Tool.TASK_LIST: "TaskList",
    Tool.TASK_UPDATE: "TaskUpdate",
    Tool.TASK_STOP: "TaskStop",
    Tool.SKILL: "Skill",
    Tool.LIST_MCP_RESOURCES: "ListMcpResourcesTool",
    Tool.READ_MCP_RESOURCE: "ReadMcpResourceTool",
    Tool.LSP: "LSP",
    Tool.ENTER_WORKTREE: "EnterWorktree",
    Tool.EXIT_WORKTREE: "ExitWorktree",
    Tool.MONITOR: "Monitor",
}

# maps an old name that the CLI's init message may still report to the name in _CLI_NAMES
_ALIASES: dict[str, str] = {"Task": "Agent"}


def _cli_names(tools: Tool) -> list[str]:
    """Converts a combination of tool flags into the CLI's tool names."""
    # iterating a Flag member yields its constituent single bits, so composites expand here
    return [_CLI_NAMES[bit] for bit in tools if bit in _CLI_NAMES]


def _builtin_of(rule: str) -> str | None:
    """Returns the built-in tool that a permission rule such as "Bash(git *)" refers to, or
    None for an MCP tool, which the CLI's list of available tools does not govern."""
    if rule.startswith("mcp__"):
        return None
    return rule.split("(", 1)[0]


def _as_loggable(message: Any) -> dict:
    """Returns an SDK message as a dict that json.dumps can write."""
    body = dataclasses.asdict(message) if dataclasses.is_dataclass(message) else {"repr": repr(message)}
    return {"type": type(message).__name__, **body}


class Agent:
    """Holds one agent's identity and permissions, and runs it on demand."""

    def __init__(
        self,
        name: str,
        profile: Path,
        *,
        cwd: Path,
        model: Model | str = Model.SONNET,
        allowed_tools: Tool = Tool.INSPECT,
        extra_allowed_tools: tuple[str, ...] = (),
        permission_mode: PermissionMode | str = PermissionMode.ACCEPT_EDITS,
        schema: dict | None = None,
        max_work_usd: float | None = None,
        timeout_s: int | None = None,
        effort: Effort | str | None = None,
        max_turns: int | None = None,
        fallback_model: str | None = None,
        clean_context: bool = True,
        log_dir: Path | None = None,
        session_id: str | None = None,
        cli_path: Path | None = None,
        on_message: Callable[[Any], None] | None = None,
        permission_policy: Callable[[str, dict], bool | tuple[bool, str]] | None = None,
    ) -> None:
        """Validates the configuration and adopts any session id handed in.

        A 'cli_path' of None lets the SDK find the CLI, first in its own bundled copy and then
        on PATH. A missing CLI is only discovered on the first run, where it raises AgentError.

        Given 'on_message', every turn calls it with each raw SDK message as the stream yields
        it, such as an AssistantMessage or the final ResultMessage. It is called synchronously
        and never awaited, so it must not be a coroutine function. Any exception it raises is
        ignored, and the turn's result is unaffected.

        Given a 'permission_policy', each tool call is passed to it as the tool's name and its
        input dict. The call runs if the policy returns True or (True, reason), and is refused
        if it returns False or (False, reason), with the refusal listed in the result's
        "denials". The built-in tools that 'allowed_tools' names are then made available
        without being approved in advance, so the policy sees every call to them. A bare tool
        name in 'extra_allowed_tools', such as "Bash" rather than "Bash(git *)", is still
        approved before the policy is asked, and so is a Bash command that the CLI treats as
        read-only, such as `echo hi`.
        """
        # a bad profile path is a caller mistake, so fail before spawning anything
        if not profile.is_file():
            raise AgentError(
                f"agent {name!r}: no profile at {profile}. "
                "profile must be a markdown file holding the agent's instructions."
            )
        if cli_path is not None and not Path(cli_path).is_file():
            raise AgentError(f"agent {name!r}: no claude executable at {cli_path}.")
        if permission_policy is not None and str(permission_mode) == str(PermissionMode.BYPASS):
            raise AgentError(
                f"agent {name!r}: permission_policy cannot be combined with "
                "PermissionMode.BYPASS, which skips the policy entirely."
            )

        self.name = name
        self.profile = profile
        self.cwd = cwd
        self.model = model
        self.allowed_tools = allowed_tools
        self.extra_allowed_tools = extra_allowed_tools
        self.permission_mode = permission_mode
        self.schema = schema
        self.max_work_usd = max_work_usd
        self.timeout_s = timeout_s
        self.effort = effort
        self.max_turns = max_turns
        self.fallback_model = fallback_model
        self.clean_context = clean_context
        self.log_dir = log_dir
        self.cli_path = cli_path
        self.on_message = on_message
        self.permission_policy = permission_policy

        # The CLI mints the id and reports it on the first reply. So None means this agent
        # has never run, and an id can only have come from a call that completed, which is
        # what lets a failed resume mean "expired" rather than "not created yet".
        self.session_id = session_id

        self.results: list[dict] = []
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    @property
    def last(self) -> dict | None:
        """Returns the most recent result, or None if this agent has not run yet."""
        return self.results[-1] if self.results else None

    def run(self, prompt: str) -> dict:
        """Runs a turn in this agent's own session, so it remembers its earlier turns.

        Blocks until the turn ends. It cannot be called from inside a running event loop,
        where arun() is the one to await instead.
        """
        return asyncio.run(self.arun(prompt))

    def run_fresh(self, prompt: str) -> dict:
        """Runs a throwaway turn with no memory, leaving this agent's session untouched."""
        return asyncio.run(self.arun_fresh(prompt))

    async def arun(self, prompt: str) -> dict:
        """Runs a turn in this agent's own session, as run() does, for async callers."""
        result = await self._invoke(prompt, continuity=True)
        self._record(result)
        # adopt whatever session the CLI used, so every later turn rejoins this conversation
        if result["ok"] and result["session_id"]:
            self.session_id = result["session_id"]
        return result

    async def arun_fresh(self, prompt: str) -> dict:
        """Runs a throwaway turn, as run_fresh() does, for async callers."""
        result = await self._invoke(prompt, continuity=False)
        self._record(result)
        return result

    def run_forked(self, prompt: str) -> dict:
        """Runs arun_forked() synchronously, as run() does for arun()."""
        return asyncio.run(self.arun_forked(prompt))

    async def arun_forked(self, prompt: str) -> dict:
        """Asks a side question against a copy of this agent's memory, without moving its
        own session forward.

        Requires a prior run() to fork from, and returns a failed result if there is none.
        The forked session's id comes back in the result's "session_id" and is not kept.
        """
        if not self.session_id:
            return self._failed("run_forked requires a prior run() to fork from")
        result = await self._invoke(prompt, continuity=False, fork=True)
        self._record(result)
        return result

    def _record(self, result: dict) -> None:
        """Files a result and adds its usage to this agent's running totals."""
        self.results.append(result)
        self.total_cost_usd += result["cost_usd"]
        self.total_tokens += result["total_tokens"]

    def reset(self) -> None:
        """Forgets the session so the next run starts a fresh conversation."""
        self.session_id = None

    def _options(
        self, *, continuity: bool, fork: bool = False, stderr: list[str] | None = None
    ) -> ClaudeAgentOptions:
        """Assembles the SDK options for one turn, collecting the CLI's stderr lines into
        'stderr' when a list is given.

        Given 'fork', the turn resumes a copy of this agent's session under a new id, and
        the session itself is left as it was.
        """
        builtin = _cli_names(self.allowed_tools)
        available = list(builtin)
        for rule in self.extra_allowed_tools:
            base = _builtin_of(rule)
            if base is not None and base not in available:
                available.append(base)

        extra_args: dict[str, str | None] = {
            # nobody is watching, so anything that would prompt is denied rather than left hanging
            "permission-prompts": "none",
            # a file path rather than the text, so a long profile cannot hit the Windows argv limit
            "append-system-prompt-file": str(self.profile.resolve()),
        }
        # safe mode drops CLAUDE.md, hooks and skills, which keeps the reply clean and the prompt small
        if self.clean_context:
            extra_args["safe-mode"] = None

        pre_approved = builtin
        can_use_tool = None
        if self.permission_policy is not None:
            policy = self.permission_policy
            # a bare name in allowed_tools is approved before can_use_tool is asked, so leave
            # the built-in names out and let the policy decide every call to them
            pre_approved = []
            # drop the flag that denies every prompt, since the policy is what answers them now
            del extra_args["permission-prompts"]

            async def can_use_tool(
                tool_name: str, tool_input: dict, context: ToolPermissionContext
            ) -> PermissionResultAllow | PermissionResultDeny:
                verdict = policy(tool_name, tool_input)
                allowed, reason = verdict if isinstance(verdict, tuple) else (verdict, None)
                if allowed:
                    return PermissionResultAllow()
                return PermissionResultDeny(message=reason or "denied by permission_policy")

        return ClaudeAgentOptions(
            model=str(self.model),
            permission_mode=str(self.permission_mode),
            # Left as None, the SDK would pass an empty system prompt and erase Claude Code's
            # own. The preset without "append" keeps it, and the profile is appended above.
            system_prompt={"type": "preset", "preset": "claude_code"},
            # tools sets what exists at all, so an empty list really does mean no tools
            tools=available,
            allowed_tools=pre_approved + list(self.extra_allowed_tools),
            can_use_tool=can_use_tool,
            output_format=(
                {"type": "json_schema", "schema": self.schema} if self.schema is not None else None
            ),
            max_budget_usd=self.max_work_usd,
            effort=str(self.effort) if self.effort is not None else None,
            max_turns=self.max_turns,
            fallback_model=self.fallback_model,
            # one rule, no state: rejoin a session we have, and let the CLI mint one when we do not
            resume=self.session_id if (continuity or fork) and self.session_id else None,
            fork_session=fork,
            cwd=self.cwd,
            cli_path=self.cli_path,
            extra_args=extra_args,
            stderr=stderr.append if stderr is not None else None,
        )

    def _session_exists(self) -> bool:
        """Answers whether the CLI still holds this agent's session on disk."""
        return get_session_info(self.session_id, directory=str(self.cwd)) is not None

    def transcript(self) -> list[SessionMessage]:
        """Returns this agent's turn-by-turn transcript: every user/assistant message,
        including tool calls and their results, in chronological order.

        Returns an empty list if this agent has never run.
        """
        if self.session_id is None:
            return []
        return get_session_messages(self.session_id, directory=str(self.cwd))

    def _write_log(
        self, options: ClaudeAgentOptions, messages: list[Any], stderr: list[str]
    ) -> Path | None:
        """Saves the options, every SDK message and the CLI's stderr, so a failed call can be
        diagnosed afterwards."""
        if self.log_dir is None:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = self.log_dir / f"{stamp}-{self.name}.log"
        settings = {
            key: value
            for key, value in vars(options).items()
            if value not in (None, [], {}, False) and not callable(value)
        }
        lines = ["--- options ---", json.dumps(settings, indent=2, default=str), "", "--- messages ---"]
        lines += [json.dumps(_as_loggable(m), default=str) for m in messages]
        lines += ["", "--- stderr ---", *stderr]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _failed(self, reason: str, log_path: Path | None = None, **extra) -> dict:
        """Builds a result describing a turn that did not complete."""
        result = {
            "ok": False,
            "text": "",
            "data": None,
            "session_id": None,
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "num_turns": 0,
            "stop_reason": None,
            "denials": [],
            "failure": reason,
            "session_expired": False,
            "log_path": log_path,
        }
        result.update(extra)
        return result

    def _expired(self, log_path: Path | None = None) -> dict:
        """Builds the result for a turn whose session has expired or been deleted.

        The turn is never retried on a fresh session. A prompt written for an agent that
        remembers is meaningless to one that does not, and it would answer anyway.
        """
        return self._failed(
            f"the session {self.session_id} has expired or was deleted",
            log_path,
            session_expired=True,
        )

    async def _invoke(self, prompt: str, *, continuity: bool, fork: bool = False) -> dict:
        """Runs the CLI once through the SDK and turns whatever came back into a result."""
        # a session that is gone costs nothing to discover, so check before spawning anything
        if (continuity or fork) and self.session_id and not self._session_exists():
            return self._expired()

        stderr: list[str] = []
        messages: list[Any] = []
        options = self._options(continuity=continuity, fork=fork, stderr=stderr)
        error: str | None = None

        try:
            async with asyncio.timeout(self.timeout_s):
                # aclosing shuts the CLI down even when the timeout abandons the stream
                async with aclosing(query(prompt=prompt, options=options)) as stream:
                    async for message in stream:
                        messages.append(message)
                        if self.on_message is not None:
                            try:
                                self.on_message(message)
                            except Exception:
                                pass
        except TimeoutError:
            error = f"timed out after {self.timeout_s}s"
        except CLINotFoundError as exc:
            raise AgentError(f"agent {self.name!r}: {exc}") from exc
        except ClaudeSDKError as exc:
            # A failed turn yields its ResultMessage before the SDK raises ResultError, so
            # the cost and usage survive in `messages` and are read below.
            detail = str(exc).strip().splitlines()
            error = detail[0][:300] if detail else type(exc).__name__

        log_path = self._write_log(options, messages, stderr)

        final = next((m for m in reversed(messages) if isinstance(m, ResultMessage)), None)
        if final is not None:
            return self._parse(final, log_path)
        # backstop for a session that vanished between the check above and the spawn
        if continuity and any("No conversation found with session ID" in line for line in stderr):
            return self._expired(log_path)
        return self._failed(error or "the CLI produced no result", log_path)

    def _parse(self, message: ResultMessage, log_path: Path | None) -> dict:
        """Turns the SDK's final ResultMessage into this library's result."""
        usage = message.usage or {}
        text = message.result or ""

        read = int(usage.get("input_tokens") or 0)
        written = int(usage.get("output_tokens") or 0)
        cached_read = int(usage.get("cache_read_input_tokens") or 0)
        cached_written = int(usage.get("cache_creation_input_tokens") or 0)

        result = {
            "ok": True,
            "text": text,
            "data": None,
            "session_id": message.session_id or None,
            "cost_usd": float(message.total_cost_usd or 0.0),
            "input_tokens": read,
            "output_tokens": written,
            "cache_read_tokens": cached_read,
            "cache_write_tokens": cached_written,
            # all four kinds, because a resumed turn's context arrives as cache reads and
            # counting only input plus output understates what the call really consumed
            "total_tokens": read + written + cached_read + cached_written,
            "num_turns": message.num_turns,
            "stop_reason": message.stop_reason,
            "denials": list(message.permission_denials or []),
            "failure": None,
            "session_expired": False,
            "log_path": log_path,
        }

        # the result itself reports a turn that did not finish, such as one over its budget
        if message.is_error:
            reason = message.subtype or message.stop_reason or "unknown"
            result["ok"] = False
            result["failure"] = f"the turn did not complete: {reason}"
            return result

        if self.schema is not None:
            if message.structured_output is not None:
                result["data"] = message.structured_output
            else:
                try:
                    result["data"] = json.loads(text)
                except json.JSONDecodeError:
                    result["ok"] = False
                    result["failure"] = "a schema was set but the reply did not parse as JSON"

        return result
