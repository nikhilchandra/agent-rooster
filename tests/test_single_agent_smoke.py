"""Proves the plumbing works, using a profile that lives on disk as a real file."""

from __future__ import annotations

from pathlib import Path

import pytest

from rooster import Agent, Tool

# the profile is a checked-in file, not written by the test, so you can open it and edit it
HELLO_PROFILE = Path(__file__).parent / "agents" / "hello.md"


@pytest.mark.live
def test_the_hello_agent_says_hello(tmp_path: Path) -> None:
    """Runs the simplest possible agent and checks it said exactly what its profile told it to."""
    agent = Agent("hello", HELLO_PROFILE, cwd=tmp_path, allowed_tools=Tool.NONE)

    result = agent.run("Go.")

    assert result["ok"], result["failure"]
    assert result["text"].strip() == "Hello!"
    print(f"\nreply: {result['text'].strip()!r}")
    print(f"cost:  ${result['cost_usd']:.4f}  tokens: {result['total_tokens']}")


def test_the_hello_profile_exists() -> None:
    """Catches a renamed or deleted profile without spending anything."""
    assert HELLO_PROFILE.is_file()
