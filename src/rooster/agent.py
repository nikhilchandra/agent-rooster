"""Runs one Claude Code agent through the Agent SDK and returns its answer as data."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shlex
from contextlib import aclosing
from datetime import datetime, timezone
from enum import Flag, StrEnum, auto
from pathlib import Path, PureWindowsPath
from typing import IO, Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLINotFoundError,
    HookMatcher,
    ResultMessage,
    get_session_info,
    query,
)
from pydantic import BaseModel, ValidationError


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


class Command(Flag):
    """Names the shell commands an agent may run, combined with the | operator.

    Tool.BASH allows every command. A Command allows only its own commands, and makes the
    Bash tool available for them.
    """

    NONE = 0
    PYTEST = auto()
    GIT_MV = auto()
    GIT_RM = auto()
    GIT_READ = auto()


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
# listed at https://code.claude.com/docs/en/tools-reference.
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

# Maps each single-bit Command to the command patterns it allows. A pattern ending in " *"
# also gets a copy without the arguments, so a bare "pytest" matches too.
_COMMAND_PATTERNS: dict[Command, tuple[str, ...]] = {
    Command.PYTEST: ("pytest *", "uv run pytest *", "uv run --no-sync pytest *", "python -m pytest *"),
    Command.GIT_MV: ("git mv *",),
    Command.GIT_RM: ("git rm *",),
    Command.GIT_READ: ("git status *", "git diff *", "git log *", "git show *", "git blame *"),
}

# the tools that change files, which only a path rule may approve once 'editable' is given
_EDIT_TOOLS = ("Edit", "Write", "NotebookEdit")

# maps an old name that the CLI's init message may still report to the name in _CLI_NAMES
_ALIASES: dict[str, str] = {"Task": "Agent"}


def _cli_names(tools: Tool) -> list[str]:
    """Converts a combination of tool flags into the CLI's tool names.

    Args:
        tools: One Tool flag, or several combined with the | operator.

    Returns:
        The CLI's name for each single tool in 'tools', such as ["Read", "Grep"].
    """
    # iterating a Flag member yields its constituent single bits, so composites expand here
    return [_CLI_NAMES[bit] for bit in tools if bit in _CLI_NAMES]


def _path_rule(pattern: str) -> str:
    """Returns a path pattern in the form that a Claude Code permission rule reads.

    A rule that starts with a single "/" is anchored at the settings file, and one that starts
    with "//" is anchored at the filesystem root. So an absolute pattern gets a second "/", and
    a Windows drive is written as a lowercase letter, which makes "C:/work/**" into
    "//c/work/**" and "/home/x" into "//home/x". A relative pattern is returned unchanged and
    stays anchored at the agent's 'cwd'.
    """
    drive = PureWindowsPath(pattern).drive
    if len(drive) == 2 and drive[1] == ":":
        return "//" + drive[0].lower() + pattern[2:].replace("\\", "/")
    if pattern.startswith("/"):
        return "/" + pattern
    return pattern


def _bash_rules(commands: Command) -> list[str]:
    """Converts a combination of command flags into the CLI's Bash permission rules.

    Args:
        commands: One Command flag, or several combined with the | operator.

    Returns:
        One rule per allowed pattern, such as ["Bash(git mv *)"].
    """
    rules = []
    for bit in commands:
        for pattern in _COMMAND_PATTERNS[bit]:
            rules.append(f"Bash({pattern})")
            if pattern.endswith(" *"):
                rules.append(f"Bash({pattern[:-2]})")
    return rules


def _as_loggable(message: Any) -> dict:
    """Returns an SDK message as a dict that json.dumps can write.

    Args:
        message: Any message that the SDK's stream yields.

    Returns:
        A dict holding the message's class name under "type", followed by its fields. A
        message that is not a dataclass is kept as its repr under "repr".
    """
    body = dataclasses.asdict(message) if dataclasses.is_dataclass(message) else {"repr": repr(message)}
    return {"type": type(message).__name__, **body}


def _forbidden_option_in(command: str, forbidden: tuple[str, ...]) -> str | None:
    """Finds the first forbidden option that a shell command uses.

    A word matches an option when it is the option itself. For a long option, which starts
    with "--", a word also matches when it is the option followed by "=" and a value, or when
    it is a shortened form that git accepts: a word of at least 4 characters that starts with
    "--" and is a prefix of the option, such as "--outp" for "--output".

    Args:
        command: The shell command, which may chain several commands.
        forbidden: The option words to look for, such as ("--no-index", "-C").

    Returns:
        The first forbidden option that a word of 'command' matches, or None when none does.
    """
    # the shell's operators, such as ";", "&&" and "|", become words of their own, so an option
    # written right against one is still a word by itself
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        words = list(lexer)
    except ValueError:
        # an unbalanced quote still gets checked, split on whitespace, rather than let through
        words = command.split()
    for word in words:
        head = word.split("=", 1)[0]
        for option in forbidden:
            if word == option:
                return option
            if not option.startswith("--"):
                continue
            if head == option or (len(head) >= 4 and head.startswith("--") and option.startswith(head)):
                return option
    return None


# the input fields of a built-in tool that name a file or folder
_PATH_FIELDS = ("file_path", "path", "notebook_path")

# the characters that make a path pattern match more than one place
_WILDCARDS = ("*", "?", "[", "{")


def _glob_folder(pattern: str) -> str:
    """Finds the folders at the start of a Glob pattern, before its first wildcard.

    Args:
        pattern: A Glob pattern, such as "src/**/*.py" or "C:/work/**/*.md".

    Returns:
        The leading folders of 'pattern', written as they are and ending in a separator, such
        as "src/" or "C:/work/". A leading "/" or drive is kept. An empty string means that the
        pattern starts with a wildcard, so it names no folder.
    """
    stops = [pattern.index(mark) for mark in _WILDCARDS if mark in pattern]
    head = pattern[: min(stops)] if stops else pattern
    cut = max(head.rfind("/"), head.rfind("\\"))
    return head[: cut + 1]


def _resolved_place(value: str, base: str) -> str:
    """Finds where a path really is, with links and Windows short names expanded.

    Args:
        value: A path as a tool call wrote it, relative to 'base' or absolute.
        base: The folder that a relative 'value' is read from.

    Returns:
        The real path, in the case form that the platform compares with.
    """
    return os.path.normcase(os.path.realpath(os.path.join(base, os.path.expanduser(value))))


def _is_within(place: str, folder: str) -> bool:
    """Tells whether a place is a folder itself or lies below it. Both are already resolved."""
    if place == folder:
        return True
    return place.startswith(folder if folder.endswith(os.sep) else folder + os.sep)


def _hidden_places(hidden: tuple[str, ...], cwd: str) -> list[str]:
    """Resolves the hidden patterns that name one folder or file, to their real places.

    A pattern counts when it holds no wildcard, or when it is such a path followed by "/**".
    Any other pattern is left to the CLI's Read deny rule, because the hook cannot tell which
    files below a folder it matches.

    Args:
        hidden: The agent's hidden path patterns.
        cwd: The folder that a relative pattern is read from.

    Returns:
        The real place of each pattern that counts.
    """
    places = []
    for pattern in hidden:
        for ending in ("/**", "\\**"):
            if pattern.endswith(ending):
                pattern = pattern[: -len(ending)]
                break
        if pattern and not any(mark in pattern for mark in _WILDCARDS):
            places.append(_resolved_place(pattern, cwd))
    return places


def _misplaced_path(
    tool_name: str, tool_input: dict, cwd: str, folder: str, places: list[str]
) -> tuple[str, str] | None:
    """Finds the first path of a tool call that lands outside a folder or in a hidden place.

    Args:
        tool_name: The tool being called. Only Glob has a pattern that is checked.
        tool_input: The tool's input. Its 'file_path', 'path' and 'notebook_path' are checked,
            and for Glob so are the leading folders of its 'pattern'. A call that names none of
            these works in 'cwd', which is inside.
        cwd: The folder that relative paths are read from.
        folder: The real place of the folder that the call must stay in.
        places: The real places that the call must stay out of.

    Returns:
        The offending path as written and the word "outside" or "hidden", or None when every
        path is fine.
    """
    checks = [
        (tool_input[field], cwd) for field in _PATH_FIELDS if isinstance(tool_input.get(field), str)
    ]
    if tool_name == "Glob" and isinstance(tool_input.get("pattern"), str):
        leading = _glob_folder(tool_input["pattern"])
        if leading:
            # a pattern is read from the 'path' field when there is one, else from cwd
            has_path = isinstance(tool_input.get("path"), str)
            checks.append((leading, _resolved_place(tool_input["path"], cwd) if has_path else cwd))
    for written, base in checks:
        place = _resolved_place(written, base)
        if not _is_within(place, folder):
            return written, "outside"
        if any(_is_within(place, hidden) for hidden in places):
            return written, "hidden"
    return None


def _pre_tool_use_hook(forbidden: tuple[str, ...], cwd: str, folder: Path | None, hidden: tuple[str, ...]):
    """Builds a PreToolUse hook that refuses a forbidden Bash option and a misplaced path.

    Args:
        forbidden: The option words that no Bash command may contain.
        cwd: The folder that the agent works in, which relative paths are read from.
        folder: A folder that file tools may not leave, or None to skip the path check.
        hidden: The agent's hidden path patterns, which the path check keeps calls out of.

    Returns:
        An async callback in the SDK's hook form. It answers with a "deny" decision that names
        the option or the path, or with an empty dict to leave the call to the other permission
        rules.
    """
    base = str(cwd)
    real_folder = _resolved_place(str(folder), base) if folder is not None else None
    places = _hidden_places(hidden, base) if folder is not None else []

    def deny(reason: str) -> dict:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    async def refuse_forbidden_options_and_misplaced_paths(input_data, tool_use_id, context) -> dict:
        tool_name = input_data["tool_name"]
        tool_input = input_data["tool_input"]
        # a Bash command is checked for forbidden options only
        if tool_name == "Bash":
            option = _forbidden_option_in(tool_input.get("command", ""), forbidden)
            if option is None:
                return {}
            return deny(f"the option {option} is forbidden here, because it reaches outside the project")
        if real_folder is None:
            return {}
        found = _misplaced_path(tool_name, tool_input, base, real_folder, places)
        if found is None:
            return {}
        path, kind = found
        if kind == "outside":
            return deny(f"the path {path} is outside {folder}, so it is refused")
        return deny(f"the path {path} is hidden from this agent, so it is refused")

    return refuse_forbidden_options_and_misplaced_paths


class Agent:
    """Holds one agent's identity and permissions, and runs it on demand.

    An agent either remembers its earlier turns or starts every turn fresh, as 'remembers'
    sets once for all its turns. run() blocks until the turn ends. arun() does the same for
    async callers, so several agents can run at once, as in
    asyncio.gather(a.arun(p), b.arun(p)). One agent must not run two turns at once, since
    both would share its one session.

    run() and arun() return a result dict with these keys:
        ok: True when the turn completed and, given a schema, its reply fit the schema.
        text: The reply's text.
        data: The reply as an instance of the schema class when a schema is set, otherwise
            None.
        session_id: The session the turn ran in, or None when the turn did not complete.
        cost_usd: The turn's cost at list price, in dollars.
        input_tokens, output_tokens, cache_read_tokens, cache_write_tokens: The turn's
            tokens of each kind.
        total_tokens: The sum of the four token counts.
        num_turns: The number of model turns the CLI took.
        stop_reason: The reason the model stopped, as the CLI reports it.
        denials: The tool calls that were refused.
        failure: The reason the turn did not complete, or None when it did.
        session_expired: True when the session to resume no longer exists.
        log_path: The turn's log file, or None when the turn never started.
    """

    def __init__(
        self,
        name: str,
        profile: Path,
        *,
        cwd: Path,
        log_dir: Path,
        remembers: bool = False,
        model: Model | str = Model.SONNET,
        effort: Effort | str | None = None,
        tools: Tool = Tool.INSPECT,
        commands: Command = Command.NONE,
        # mcp_tools: tuple[str, ...] = (),  # MCP tools such as "mcp__server__tool", not yet supported
        editable: tuple[str, ...] = (),
        frozen: tuple[str, ...] = (),
        hidden: tuple[str, ...] = (),
        forbidden_options: tuple[str, ...] = (),
        confined_to: Path | None = None,
        schema: type[BaseModel] | None = None,
        max_work_usd: float | None = None,
        timeout_s: int | None = None,
        safe_mode: bool = True,
        setting_sources: tuple[str, ...] | None = None,
        permission_mode: PermissionMode | str = PermissionMode.ACCEPT_EDITS,
        session_id: str | None = None,
        cli_path: Path | None = None,
    ) -> None:
        """Validates the configuration and adopts any session id handed in.

        Args:
            name: A short name for the agent, used in errors and log file names.
            profile: A markdown file holding the agent's instructions. It is appended to
                Claude Code's own system prompt.
            cwd: The folder that the agent works in.
            log_dir: The folder for this agent's log files, one per turn. Each file is
                written line by line while the turn runs, so it can be watched live.
            remembers: True continues one conversation across all of this agent's turns.
                False starts every turn fresh, so the prompt must carry everything the agent
                needs.
            model: A Model, or a full model ID as a plain string.
            effort: How much the model thinks, from Effort.LOW to Effort.MAX. None leaves the
                CLI's default.
            tools: The built-in tools that the agent may use.
            commands: The shell commands that the agent may run, such as
                Command.PYTEST | Command.GIT_MV. Tool.BASH in 'tools' allows every command
                instead.
            editable: Path patterns, relative to 'cwd', that the agent may change, such as
                "src/**". Empty means any file under 'cwd'. Given any pattern, the agent
                runs in PermissionMode.DONT_ASK in place of PermissionMode.ACCEPT_EDITS.
            frozen: Path patterns, relative to 'cwd', that the agent may never change, even
                inside 'editable', such as "**/*.feature".
            hidden: Path patterns that the agent may never read, relative to 'cwd' or absolute,
                such as "secrets/**" or a folder outside 'cwd'. Claude Code enforces them on
                Read, Grep and Glob. With 'confined_to', the plain and "folder/**" patterns are
                also enforced by real location.
            forbidden_options: Command-line option words that no Bash command may contain, such
                as "--no-index" or "-C". A PreToolUse hook refuses them, even on calls that the
                allow rules approve. A long option also catches its "=value" form and git's
                shortened forms, such as "--outp" for "--output".
            confined_to: A folder that file tools may not leave. A PreToolUse hook resolves each
                path that a tool call names, links and short names included, and refuses a call
                that lands outside the folder or inside a plain or "folder/**" pattern of
                'hidden'. None leaves paths to the permission rules.
            schema: A pydantic class that the reply must fit. The reply then arrives as an
                instance of it in the result's "data".
            max_work_usd: The most that one turn may spend, in dollars. None means no limit.
            timeout_s: The longest that one turn may take, in seconds. None means no limit.
            safe_mode: True runs the CLI in safe mode, which leaves out CLAUDE.md, hooks and
                skills.
            setting_sources: The Claude settings files that the CLI loads, from "user", "project"
                and "local". None leaves the CLI's default, which loads them all. An empty tuple
                loads none, so no settings file's allow rules, env or model reach the agent.
            permission_mode: The CLI's permission mode. The default lets file edits through
                without asking, unless 'editable' is given.
            session_id: A session to resume, saved from an earlier run. It is used only when
                'remembers' is True.
            cli_path: The claude executable. None lets the SDK find the CLI, first in its own
                bundled copy and then on PATH. A missing CLI is only discovered on the first
                run, where it raises AgentError.

        Raises:
            AgentError: 'profile' is not a file, 'cli_path' names no file, or 'editable' is
                combined with PermissionMode.BYPASS.
        """
        # a bad profile path is a caller mistake, so fail before spawning anything
        if not profile.is_file():
            raise AgentError(
                f"agent {name!r}: no profile at {profile}. "
                "profile must be a markdown file holding the agent's instructions."
            )
        if cli_path is not None and not Path(cli_path).is_file():
            raise AgentError(f"agent {name!r}: no claude executable at {cli_path}.")
        if editable and str(permission_mode) == str(PermissionMode.BYPASS):
            raise AgentError(
                f"agent {name!r}: editable cannot be combined with PermissionMode.BYPASS, "
                "which approves every edit before the path rules are checked."
            )

        self.name = name
        self.profile = profile
        self.cwd = cwd
        self.log_dir = log_dir
        self.remembers = remembers
        self.model = model
        self.effort = effort
        self.tools = tools
        self.commands = commands
        self.editable = editable
        self.frozen = frozen
        self.hidden = hidden
        self.forbidden_options = forbidden_options
        self.confined_to = confined_to
        self.schema = schema
        self.max_work_usd = max_work_usd
        self.timeout_s = timeout_s
        self.safe_mode = safe_mode
        self.setting_sources = setting_sources
        self.permission_mode = permission_mode
        self.cli_path = cli_path

        # The CLI mints the id and reports it on the first reply. So None means this agent
        # has never run, and an id can only have come from a call that completed, which is
        # what lets a failed resume mean "expired" rather than "not created yet".
        self.session_id = session_id

        # The cost in dollars and the tokens of every turn this agent has run.
        self.total_cost_usd = 0.0
        self.total_tokens = 0

    def run(self, prompt: str) -> dict:
        """Runs one turn and blocks until it ends.

        It cannot be called from inside a running event loop, where arun() is the one to
        await instead.

        Args:
            prompt: The message to send.

        Returns:
            The turn's result dict, with the keys that the Agent class docstring lists.
        """
        return asyncio.run(self.arun(prompt))

    async def arun(self, prompt: str) -> dict:
        """Runs one turn, as run() does, for async callers.

        Args:
            prompt: The message to send.

        Returns:
            The turn's result dict, with the keys that the Agent class docstring lists.
        """
        result = await self._invoke(prompt)
        self.total_cost_usd += result["cost_usd"]
        self.total_tokens += result["total_tokens"]
        # adopt whatever session the CLI used, so every later turn rejoins this conversation
        if self.remembers and result["ok"] and result["session_id"]:
            self.session_id = result["session_id"]
        return result

    def _options(self, *, stderr: list[str] | None = None) -> ClaudeAgentOptions:
        """Assembles the SDK options for one turn.

        Args:
            stderr: A list that collects the CLI's stderr lines. None discards them.

        Returns:
            The options to pass to the SDK's query().
        """
        builtin = _cli_names(self.tools)
        bash_rules = _bash_rules(self.commands)
        # tools sets what exists at all, so a command rule needs Bash to exist
        available = builtin + (["Bash"] if bash_rules and "Bash" not in builtin else [])

        pre_approved = builtin
        path_rules: list[str] = []
        permission_mode = str(self.permission_mode)
        if self.editable:
            # the bare edit tools would approve every path, so only the path rules approve edits
            pre_approved = [name for name in builtin if name not in _EDIT_TOOLS]
            path_rules = [f"Edit({pattern})" for pattern in self.editable]
            # accept-edits mode approves every edit before the path rules are checked
            if permission_mode == str(PermissionMode.ACCEPT_EDITS):
                permission_mode = str(PermissionMode.DONT_ASK)

        extra_args: dict[str, str | None] = {
            # nobody is watching, so anything that would prompt is denied rather than left hanging
            "permission-prompts": "none",
            # a file path rather than the text, so a long profile cannot hit the Windows argv limit
            "append-system-prompt-file": str(self.profile.resolve()),
        }
        # safe mode drops CLAUDE.md, hooks and skills, which keeps the reply clean and the prompt small
        if self.safe_mode:
            extra_args["safe-mode"] = None

        return ClaudeAgentOptions(
            model=str(self.model),
            permission_mode=permission_mode,
            # Left as None, the SDK would pass an empty system prompt and erase Claude Code's
            # own. The preset without "append" keeps it, and the profile is appended above.
            system_prompt={"type": "preset", "preset": "claude_code"},
            # tools sets what exists at all, so an empty list really does mean no tools
            tools=available,
            allowed_tools=pre_approved + bash_rules + path_rules,
            # deny rules are checked before anything else, so a frozen or hidden path holds in every mode
            # a Read rule also covers Grep and Glob, which have no path rules of their own
            disallowed_tools=[f"Edit({pattern})" for pattern in self.frozen]
            + [f"Read({_path_rule(pattern)})" for pattern in self.hidden],
            output_format=(
                {"type": "json_schema", "schema": self.schema.model_json_schema()}
                if self.schema is not None
                else None
            ),
            max_budget_usd=self.max_work_usd,
            effort=str(self.effort) if self.effort is not None else None,
            # one rule, no state: rejoin a session we have, and let the CLI mint one when we do not
            resume=self.session_id if self.remembers and self.session_id else None,
            # a hook runs on every call, but an allow rule approves a call before can_use_tool is asked
            hooks=(
                {
                    "PreToolUse": [
                        HookMatcher(
                            matcher=None if self.confined_to is not None else "Bash",
                            hooks=[
                                _pre_tool_use_hook(
                                    self.forbidden_options, self.cwd, self.confined_to, self.hidden
                                )
                            ],
                        )
                    ]
                }
                if self.forbidden_options or self.confined_to is not None
                else None
            ),
            setting_sources=(
                list(self.setting_sources) if self.setting_sources is not None else None
            ),
            cwd=self.cwd,
            cli_path=self.cli_path,
            extra_args=extra_args,
            stderr=stderr.append if stderr is not None else None,
        )

    def _open_log(self, options: ClaudeAgentOptions, prompt: str) -> tuple[Path, IO[str]]:
        """Creates this turn's log file and writes the options and the prompt it runs with.

        The file has four sections: the options, the prompt exactly as sent, then one JSON
        line per SDK message as it arrives, then the CLI's stderr once the turn ends.

        Args:
            options: The options that the turn runs with.
            prompt: The message that the turn sends.

        Returns:
            The log file's path, and the file opened for the messages that follow.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        path = self.log_dir / f"{stamp}-{self.name}.log"
        settings = {
            key: value
            for key, value in vars(options).items()
            if value not in (None, [], {}, False) and not callable(value)
        }
        log = path.open("w", encoding="utf-8")
        log.write("--- options ---\n" + json.dumps(settings, indent=2, default=str) + "\n\n")
        # the SDK never echoes the prompt back among the messages, so this is its only record
        log.write("--- prompt ---\n" + prompt + "\n\n")
        log.write("--- messages ---\n")
        log.flush()
        return path, log

    def _failed(self, reason: str, log_path: Path | None = None, **extra) -> dict:
        """Builds a result describing a turn that did not complete.

        Args:
            reason: The reason the turn did not complete, stored as "failure".
            log_path: The turn's log file, or None when the turn never started.
            **extra: Result entries that replace the defaults, such as session_expired=True.

        Returns:
            A result dict whose "ok" is False and whose counts are all zero.
        """
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

    async def _invoke(self, prompt: str) -> dict:
        """Runs the CLI once through the SDK and turns whatever came back into a result.

        A turn whose session has expired is never retried on a fresh session. A prompt
        written for an agent that remembers is meaningless to one that does not, and it
        would answer anyway.

        Args:
            prompt: The message to send.

        Returns:
            The turn's result dict, with the keys that the Agent class docstring lists.

        Raises:
            AgentError: The claude CLI cannot be found.
        """
        expired = f"the session {self.session_id} has expired or was deleted"
        # a session that is gone costs nothing to discover, so check before spawning anything
        if (
            self.remembers
            and self.session_id
            and get_session_info(self.session_id, directory=str(self.cwd)) is None
        ):
            return self._failed(expired, session_expired=True)

        stderr: list[str] = []
        messages: list[Any] = []
        options = self._options(stderr=stderr)
        error: str | None = None
        log_path, log = self._open_log(options, prompt)

        try:
            async with asyncio.timeout(self.timeout_s):
                # aclosing shuts the CLI down even when the timeout abandons the stream
                async with aclosing(query(prompt=prompt, options=options)) as stream:
                    async for message in stream:
                        messages.append(message)
                        log.write(json.dumps(_as_loggable(message), default=str) + "\n")
                        log.flush()
        except TimeoutError:
            error = f"timed out after {self.timeout_s}s"
        except CLINotFoundError as exc:
            raise AgentError(f"agent {self.name!r}: {exc}") from exc
        except ClaudeSDKError as exc:
            # A failed turn yields its ResultMessage before the SDK raises ResultError, so
            # the cost and usage survive in `messages` and are read below.
            detail = str(exc).strip().splitlines()
            error = detail[0][:300] if detail else type(exc).__name__
        finally:
            log.write("\n--- stderr ---\n" + "".join(line + "\n" for line in stderr))
            log.close()

        final = next((m for m in reversed(messages) if isinstance(m, ResultMessage)), None)
        if final is not None:
            return self._parse(final, log_path)
        # backstop for a session that vanished between the check above and the spawn
        if self.remembers and any("No conversation found with session ID" in line for line in stderr):
            return self._failed(expired, log_path, session_expired=True)
        return self._failed(error or "the CLI produced no result", log_path)

    def _parse(self, message: ResultMessage, log_path: Path | None) -> dict:
        """Turns the SDK's final ResultMessage into this library's result.

        Args:
            message: The SDK's final ResultMessage for the turn.
            log_path: The turn's log file, or None when the turn never started.

        Returns:
            The turn's result dict. Its "ok" is False when the turn reports an error, or when
            a schema was set and the reply does not fit it.
        """
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
            try:
                if message.structured_output is not None:
                    result["data"] = self.schema.model_validate(message.structured_output)
                else:
                    result["data"] = self.schema.model_validate_json(text)
            except ValidationError as exc:
                result["ok"] = False
                result["failure"] = f"the reply does not fit {self.schema.__name__}: {exc.errors()[0]['msg']}"

        return result
