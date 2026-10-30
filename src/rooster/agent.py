"""Runs one Claude Code agent as a subprocess and returns its answer as data."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from enum import Flag, StrEnum, auto
from pathlib import Path


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
        denied_tools: Tool = Tool.NONE,
        extra_allowed_tools: tuple[str, ...] = (),
        permission_mode: PermissionMode | str = PermissionMode.ACCEPT_EDITS,
        schema: dict | None = None,
        max_work_usd: float | None = None,
        timeout_s: int | None = None,
        clean_context: bool = True,
        log_dir: Path | None = None,
        session_id: str | None = None,
    ) -> None:
        """Validates the configuration and mints this agent's session id."""
        # a bad profile path or a missing binary is a caller mistake, so fail before spawning anything
        if not profile.is_file():
            raise AgentError(
                f"agent {name!r}: no profile at {profile}. "
                "profile must be a markdown file holding the agent's instructions."
            )
        exe = shutil.which("claude")
        if exe is None:
            raise AgentError(
                f"agent {name!r}: the 'claude' executable is not on PATH. "
                "install Claude Code, or add its directory to PATH."
            )

        self.name = name
        self.profile = profile
        self.cwd = cwd
        self.model = model
        self.allowed_tools = allowed_tools
        self.denied_tools = denied_tools
        self.extra_allowed_tools = extra_allowed_tools
        self.permission_mode = permission_mode
        self.schema = schema
        self.max_work_usd = max_work_usd
        self.timeout_s = timeout_s
        self.clean_context = clean_context
        self.log_dir = log_dir

        self._exe = exe
        # the id is minted up front so logs can be named after it before the first call returns
        self.session_id = session_id or str(uuid.uuid4())
        self._started = False

        self.results: list[dict] = []
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    @property
    def last(self) -> dict | None:
        """Returns the most recent result, or None if this agent has not run yet."""
        return self.results[-1] if self.results else None

    def run(self, prompt: str) -> dict:
        """Runs a turn in this agent's own session, so it remembers its earlier turns."""
        result = self._invoke(prompt, continuity=True)
        self._record(result)
        # only mark the session live once a turn has actually produced one
        if result["ok"] and result["session_id"]:
            self._started = True
        return result

    def run_fresh(self, prompt: str) -> dict:
        """Runs a throwaway turn with no memory, leaving this agent's session untouched."""
        result = self._invoke(prompt, continuity=False)
        self._record(result)
        return result

    def _record(self, result: dict) -> None:
        """Files a result and adds its usage to this agent's running totals."""
        self.results.append(result)
        self.total_cost_usd += result["cost_usd"]
        self.total_tokens += result["total_tokens"]

    def reset(self) -> None:
        """Mints a new session id so the next run starts a fresh conversation."""
        self.session_id = str(uuid.uuid4())
        self._started = False

    def _argv(self, *, continuity: bool) -> list[str]:
        """Assembles the command line for one turn."""
        argv = [
            self._exe,
            "--print",
            "--output-format",
            "json",
            "--model",
            str(self.model),
            "--permission-mode",
            str(self.permission_mode),
            # nobody is watching, so anything that would prompt is denied rather than left hanging
            "--permission-prompts",
            "none",
            "--append-system-prompt-file",
            str(self.profile),
        ]

        # safe mode drops CLAUDE.md, hooks and skills, which keeps the reply clean and the prompt small
        if self.clean_context:
            argv.append("--safe-mode")

        allowed = _cli_names(self.allowed_tools) + list(self.extra_allowed_tools)
        denied = _cli_names(self.denied_tools)
        if allowed:
            argv += ["--allowed-tools", *allowed]
        else:
            # an empty allowlist must mean no tools; omitting the flag would instead mean all of them
            denied = sorted(set(denied) | set(_CLI_NAMES.values()))
        if denied:
            argv += ["--disallowed-tools", *denied]

        if self.schema is not None:
            argv += ["--json-schema", json.dumps(self.schema)]
        if self.max_work_usd is not None:
            argv += ["--max-budget-usd", str(self.max_work_usd)]

        if continuity:
            # the first turn creates the session under our id; later turns rejoin it
            argv += ["--resume", self.session_id] if self._started else ["--session-id", self.session_id]

        return argv

    def _write_log(self, argv: list[str], stdout: str, stderr: str) -> Path | None:
        """Saves the raw exchange so a failed call can be diagnosed afterwards."""
        if self.log_dir is None:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = self.log_dir / f"{stamp}-{self.name}.log"
        # the command line goes first, because a malformed flag is the most common cause of a bad call
        path.write_text(
            f"$ {' '.join(argv)}\n\n--- stdout ---\n{stdout}\n\n--- stderr ---\n{stderr}\n",
            encoding="utf-8",
        )
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
            "log_path": log_path,
        }
        result.update(extra)
        return result

    def _invoke(self, prompt: str, *, continuity: bool) -> dict:
        """Runs the CLI once and turns whatever came back into a result."""
        argv = self._argv(continuity=continuity)

        try:
            proc = subprocess.run(
                argv,
                # the prompt travels on stdin, so a long feature file cannot hit the Windows argv limit
                input=prompt,
                capture_output=True,
                text=True,
                # Windows would otherwise decode as cp1252 and mangle every dash and quote
                encoding="utf-8",
                errors="replace",
                cwd=str(self.cwd),
                shell=False,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            return self._failed(f"timed out after {self.timeout_s}s")

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        log_path = self._write_log(argv, stdout, stderr)

        # the CLI itself failed: a bad flag, an auth problem, or a crash
        if proc.returncode != 0:
            detail = (stderr.strip() or stdout.strip()).splitlines()
            first = detail[0][:300] if detail else "no output"
            return self._failed(f"exit {proc.returncode}: {first}", log_path)

        if not stdout.strip():
            return self._failed("the CLI produced no output", log_path)

        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            # an error message printed as plain text lands here, so surface its first line
            first = stdout.strip().splitlines()[0][:300]
            return self._failed(f"output was not the JSON envelope: {first}", log_path)

        return self._parse(envelope, log_path)

    def _parse(self, envelope: dict, log_path: Path | None) -> dict:
        """Turns the CLI's JSON envelope into this library's result."""
        usage = envelope.get("usage") or {}
        text = envelope.get("result") or ""

        read = int(usage.get("input_tokens") or 0)
        written = int(usage.get("output_tokens") or 0)
        cached_read = int(usage.get("cache_read_input_tokens") or 0)
        cached_written = int(usage.get("cache_creation_input_tokens") or 0)

        result = {
            "ok": True,
            "text": text,
            "data": None,
            "session_id": envelope.get("session_id"),
            "cost_usd": float(envelope.get("total_cost_usd") or 0.0),
            "input_tokens": read,
            "output_tokens": written,
            "cache_read_tokens": cached_read,
            "cache_write_tokens": cached_written,
            # all four kinds, because a resumed turn's context arrives as cache reads and
            # counting only input plus output understates what the call really consumed
            "total_tokens": read + written + cached_read + cached_written,
            "num_turns": int(envelope.get("num_turns") or 0),
            "stop_reason": envelope.get("stop_reason"),
            "denials": envelope.get("permission_denials") or [],
            "failure": None,
            "log_path": log_path,
        }

        # the CLI ran cleanly but the turn did not finish, which is a separate signal from the exit code
        if envelope.get("is_error", True):
            reason = envelope.get("subtype") or envelope.get("stop_reason") or "unknown"
            result["ok"] = False
            result["failure"] = f"the turn did not complete: {reason}"
            return result

        if self.schema is not None:
            try:
                result["data"] = json.loads(text)
            except json.JSONDecodeError:
                result["ok"] = False
                result["failure"] = "a schema was set but the reply did not parse as JSON"

        return result
