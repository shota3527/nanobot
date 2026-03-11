from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.tool_arguments import expand_clipboard_references
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path: Path) -> AgentLoop:
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model", memory_window=10)


def test_expand_clipboard_references_replaces_nested_strings(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "note.txt").write_text("hello\nworld", encoding="utf-8")

    arguments = {
        "path": "docs/{{@clip:notes/note.txt}}.md",
        "nested": ["prefix {{@clip:notes/note.txt}} suffix", {"text": "{{@clip:notes/note.txt}}"}],
        "count": 3,
    }

    expanded = expand_clipboard_references(arguments, tmp_path)

    assert expanded == {
        "path": "docs/hello\nworld.md",
        "nested": ["prefix hello\nworld suffix", {"text": "hello\nworld"}],
        "count": 3,
    }


def test_expand_clipboard_references_supports_workspace_root_shorthand(tmp_path: Path) -> None:
    (tmp_path / "ROOT.txt").write_text("root content", encoding="utf-8")

    expanded = expand_clipboard_references(
        {"text": "Value: {{@clip:/ROOT.txt}}"},
        tmp_path,
    )

    assert expanded == {"text": "Value: root content"}


def test_expand_clipboard_references_allows_absolute_path_when_unrestricted(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}_external.txt"
    external.write_text("external", encoding="utf-8")
    try:
        expanded = expand_clipboard_references(
            {"text": f"{{{{@clip:{external}}}}}"},
            tmp_path,
        )
        assert expanded == {"text": "external"}
    finally:
        external.unlink(missing_ok=True)


def test_expand_clipboard_references_blocks_outside_workspace_when_restricted(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}_blocked.txt"
    external.write_text("blocked", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside allowed directory"):
            expand_clipboard_references(
                {"text": f"{{{{@clip:{external}}}}}"},
                tmp_path,
                restrict_to_workspace=True,
            )
    finally:
        external.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_run_agent_loop_expands_clipboard_references_before_tool_execute(tmp_path: Path) -> None:
    snippets_dir = tmp_path / "snippets"
    snippets_dir.mkdir()
    (snippets_dir / "snippet.txt").write_text("expanded text", encoding="utf-8")

    loop = _make_loop(tmp_path)
    tool_call = ToolCallRequest(
        id="call1",
        name="write_file",
        arguments={"path": "out.txt", "content": "Before {{@clip:snippets/snippet.txt}} after"},
    )
    calls = iter([
        LLMResponse(content="", tool_calls=[tool_call]),
        LLMResponse(content="Done", tool_calls=[]),
    ])
    loop.provider.chat = AsyncMock(side_effect=lambda *a, **kw: next(calls))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(return_value="ok")

    final_content, _, messages = await loop._run_agent_loop([])

    assert final_content == "Done"
    loop.tools.execute.assert_awaited_once_with(
        "write_file",
        {"path": "out.txt", "content": "Before expanded text after"},
    )
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "out.txt", "content": "Before expanded text after"}'
    )
