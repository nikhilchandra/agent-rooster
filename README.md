# agent-rooster

Infrastructure for building agentic workflows on top of the Claude Code CLI.

Rooster gives you a way to define an agent, run it as a subprocess, and get its
answer back as data a Python script can branch on. It is a library, not a workflow:
it ships no roles, no pipeline and no opinion about how you develop software. You
define your agents; rooster runs them.

Because the runtime is the `claude` CLI, an agent inherits tool use, permissions,
MCP servers and file editing for free, and runs on a flat-rate Claude subscription
rather than metered API tokens.

## Install

```bash
uv tool install ~/code/agent-rooster
```

## Status

`Agent` works. `AgentGraph`, the CLI builder and the provisioning helpers are
designed but not yet written.
