"""Runs one Claude Code agent through the Agent SDK and returns its answer as data."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from contextlib import aclosing
from datetime import datetime, timezone
from enum import Flag, StrEnum, auto
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLINotFoundError,
    ResultMessage,
    get_session_info,
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
    TASK = auto()
    TODO = auto()
    WEB_FETCH = auto()
    WEB_SEARCH = auto()

    # composites, so a caller writes the intent rather than assembling it every time
    INSPECT = READ | GLOB | GREP
    AUTHOR = READ | GLOB | GREP | WRITE | EDIT
    FULL = READ | GLOB | GREP | WRITE | EDIT | BASH | TASK | TODO | WEB_FETCH | WEB_SEARCH


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


# maps each single-bit Tool to the name the CLI knows it by
_CLI_NAMES: dict[Tool, str] = {
    Tool.READ: "Read",
    Tool.WRITE: "Write",
    Tool.EDIT: "Edit",
    Tool.BASH: "Bash",
    Tool.GLOB: "Glob",
    Tool.GREP: "Grep",
    Tool.TASK: "Task",
    Tool.TODO: "TodoWrite",
    Tool.WEB_FETCH: "WebFetch",
    Tool.WEB_SEARCH: "WebSearch",
}


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
        clean_context: bool = True,
        log_dir: Path | None = None,
        session_id: str | None = None,
        cli_path: Path | None = None,
    ) -> None:
        """Validates the configuration and adopts any session id handed in.

        A 'cli_path' of None lets the SDK find the CLI, first in its own bundled copy and then
        on PATH. A missing CLI is only discovered on the first run, where it raises AgentError.
        """
        # a bad profile path is a caller mistake, so fail before spawning anything
        if not profile.is_file():
            raise AgentError(
                f"agent {name!r}: no profile at {profile}. "
                "profile must be a markdown file holding the agent's instructions."
            )
        if cli_path is not None and not Path(cli_path).is_file():
            raise AgentError(f"agent {name!r}: no claude executable at {cli_path}.")

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
        self.clean_context = clean_context
        self.log_dir = log_dir
        self.cli_path = cli_path

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

    def _record(self, result: dict) -> None:
        """Files a result and adds its usage to this agent's running totals."""
        self.results.append(result)
        self.total_cost_usd += result["cost_usd"]
        self.total_tokens += result["total_tokens"]

    def reset(self) -> None:
        """Forgets the session so the next run starts a fresh conversation."""
        self.session_id = None

    def _options(self, *, continuity: bool, stderr: list[str] | None = None) -> ClaudeAgentOptions:
        """Assembles the SDK options for one turn, collecting the CLI's stderr lines into
        'stderr' when a list is given."""
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

        return ClaudeAgentOptions(
            model=str(self.model),
            permission_mode=str(self.permission_mode),
            # Left as None, the SDK would pass an empty system prompt and erase Claude Code's
            # own. The preset without "append" keeps it, and the profile is appended above.
            system_prompt={"type": "preset", "preset": "claude_code"},
            # tools sets what exists at all, so an empty list really does mean no tools
            tools=available,
            allowed_tools=builtin + list(self.extra_allowed_tools),
            output_format=(
                {"type": "json_schema", "schema": self.schema} if self.schema is not None else None
            ),
            max_budget_usd=self.max_work_usd,
            # one rule, no state: rejoin a session we have, and let the CLI mint one when we do not
            resume=self.session_id if continuity and self.session_id else None,
            cwd=self.cwd,
            cli_path=self.cli_path,
            extra_args=extra_args,
            stderr=stderr.append if stderr is not None else None,
        )

    def _session_exists(self) -> bool:
        """Answers whether the CLI still holds this agent's session on disk."""
        return get_session_info(self.session_id, directory=str(self.cwd)) is not None

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

    async def _invoke(self, prompt: str, *, continuity: bool) -> dict:
        """Runs the CLI once through the SDK and turns whatever came back into a result."""
        # a session that is gone costs nothing to discover, so check before spawning anything
        if continuity and self.session_id and not self._session_exists():
            return self._expired()

        stderr: list[str] = []
        messages: list[Any] = []
        options = self._options(continuity=continuity, stderr=stderr)
        error: str | None = None

        try:
            async with asyncio.timeout(self.timeout_s):
                # aclosing shuts the CLI down even when the timeout abandons the stream
                async with aclosing(query(prompt=prompt, options=options)) as stream:
                    async for message in stream:
                        messages.append(message)
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
