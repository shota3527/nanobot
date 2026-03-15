"""Tests for cache-friendly prompt construction."""

from __future__ import annotations

from datetime import datetime as real_datetime
from importlib.resources import files as pkg_files
from pathlib import Path
import datetime as datetime_module

from nanobot.agent.context import ContextBuilder


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 2, 24, 13, 59)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def test_bootstrap_files_are_backed_by_templates() -> None:
    template_dir = pkg_files("nanobot") / "templates"

    for filename in ContextBuilder.BOOTSTRAP_FILES:
        assert (template_dir / filename).is_file(), f"missing bootstrap template: {filename}"


def test_system_prompt_stays_stable_when_clock_changes(tmp_path, monkeypatch) -> None:
    """System prompt should not change just because wall clock minute changes."""
    monkeypatch.setattr(datetime_module, "datetime", _FakeDatetime)

    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    _FakeDatetime.current = real_datetime(2026, 2, 24, 13, 59)
    prompt1 = builder.build_system_prompt()

    _FakeDatetime.current = real_datetime(2026, 2, 24, 14, 0)
    prompt2 = builder.build_system_prompt()

    assert prompt1 == prompt2


def test_runtime_context_is_separate_untrusted_user_message(tmp_path) -> None:
    """Runtime metadata should be merged with the user message."""
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace)

    messages = builder.build_messages(
        history=[],
        current_message="Return exactly: OK",
        channel="cli",
        chat_id="direct",
    )

    assert messages[0]["role"] == "system"
    assert "## Current Session" not in messages[0]["content"]

    # Runtime context is now merged with user message into a single message
    assert messages[-1]["role"] == "user"
    user_content = messages[-1]["content"]
    assert isinstance(user_content, str)
    assert ContextBuilder._RUNTIME_CONTEXT_TAG in user_content
    assert "Current Time:" in user_content
    assert "Channel: cli" in user_content
    assert "Chat ID: direct" in user_content
    assert "Return exactly: OK" in user_content


def test_add_tool_result_truncates_middle_to_tool_limit(tmp_path) -> None:
    workspace = _make_workspace(tmp_path)
    builder = ContextBuilder(workspace, tool_result_max_bytes=120)
    messages: list[dict[str, str]] = []
    content = "A" * 80 + "B" * 80

    builder.add_tool_result(messages, "call_1", "read_file", content)

    stored = messages[0]["content"]
    assert len(stored.encode("utf-8")) > 120
    assert "Tool output truncated in the middle" in stored
    assert stored.startswith("A" * 60)
    assert stored.endswith("B" * 60)
