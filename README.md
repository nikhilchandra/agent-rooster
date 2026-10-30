# agent-rooster

Infrastructure for building agentic workflows on top of the Claude Code CLI.

Rooster gives you a way to define an agent, run it as a subprocess, and get its
answer back as data a Python script can branch on. It is a library, not a workflow:
it ships no roles, no pipeline and no opinion about how you develop software. You
define your agents; rooster runs them.

An agent runs through Anthropic's [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python),
which drives the `claude` CLI. So an agent inherits tool use, permissions, MCP servers
and file editing, and runs against your Claude plan. What rooster adds on top is
`AgentGraph`: multi-agent runs that are saved to disk, can stop at a gate for a human,
and resume later from a fresh process.

## Install

```bash
uv tool install ~/code/agent-rooster
```

## Status

`Agent` and `AgentGraph` work. The CLI builder and the provisioning helpers are
designed but not yet written.
