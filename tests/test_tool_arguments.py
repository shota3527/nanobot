from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.clip_references import ClipboardExpansionErrorArgs, ClipboardReferenceExpander
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_loop(tmp_path: Path) -> AgentLoop:
    from nanobot.bus.queue import MessageBus
    from nanobot.providers.base import GenerationSettings

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=0)
    provider.chat_with_retry = AsyncMock(return_value=LLMResponse(content="Done", tool_calls=[]))
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content="Done", tool_calls=[]))
    return AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")


def test_clipboard_reference_expander_replaces_nested_strings(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "note.txt").write_text("hello\nworld", encoding="utf-8")

    arguments = {
        "path": "docs\n{@clip:notes/note.txt}\n.md",
        "nested": ["prefix\n{@clip:notes/note.txt}\nsuffix", {"text": "{@clip:notes/note.txt}"}],
        "count": 3,
    }

    expanded = ClipboardReferenceExpander(workspace=tmp_path).expand(arguments)

    assert expanded == {
        "path": "docs\nhello\nworld\n.md",
        "nested": ["prefix\nhello\nworld\nsuffix", {"text": "hello\nworld"}],
        "count": 3,
    }


def test_clipboard_reference_expander_supports_workspace_root_shorthand(tmp_path: Path) -> None:
    (tmp_path / "ROOT.txt").write_text("root content", encoding="utf-8")

    expanded = ClipboardReferenceExpander(workspace=tmp_path).expand(
        {"text": "Value\n{@clip:/ROOT.txt}"}
    )

    assert expanded == {"text": "Value\nroot content"}


def test_clipboard_reference_expander_allows_absolute_path_when_unrestricted(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}_external.txt"
    external.write_text("external", encoding="utf-8")
    try:
        expanded = ClipboardReferenceExpander(workspace=tmp_path).expand(
            {"text": f"{{@clip:{external}}}"}
        )
        assert expanded == {"text": "external"}
    finally:
        external.unlink(missing_ok=True)


def test_clipboard_reference_expander_prefers_relative_path_before_absolute_path(tmp_path: Path) -> None:
    relative_target = tmp_path / "tmp" / "nanobot-tool-arguments-clip.txt"
    relative_target.parent.mkdir()
    relative_target.write_text("workspace copy", encoding="utf-8")

    absolute_target = Path("/tmp/nanobot-tool-arguments-clip.txt")
    absolute_target.write_text("absolute copy", encoding="utf-8")
    try:
        expanded = ClipboardReferenceExpander(workspace=tmp_path).expand(
            {"text": "{@clip:/tmp/nanobot-tool-arguments-clip.txt}"}
        )
        assert expanded == {"text": "workspace copy"}
    finally:
        absolute_target.unlink(missing_ok=True)


def test_clipboard_reference_expander_ignores_leading_work_dir_when_needed(tmp_path: Path) -> None:
    workspace = tmp_path / "work_dir"
    workspace.mkdir()
    (workspace / "note.txt").write_text("inside work_dir", encoding="utf-8")

    expanded = ClipboardReferenceExpander(workspace=workspace).expand(
        {"text": "{@clip:work_dir/note.txt}"}
    )

    assert expanded == {"text": "inside work_dir"}


def test_clipboard_reference_expander_tolerates_spaces_inside_marker(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("spaced marker", encoding="utf-8")

    expanded = ClipboardReferenceExpander(workspace=tmp_path).expand(
        {"text": "{ @clip: note.txt }"}
    )

    assert expanded == {"text": "spaced marker"}


def test_expand_response_wraps_failed_tool_call_without_changing_tool_name(tmp_path: Path) -> None:
    expander = ClipboardReferenceExpander(workspace=tmp_path)
    response = LLMResponse(
        content="note\n{@clip:missing-message.txt}",
        tool_calls=[
            ToolCallRequest(
                id="call1",
                name="write_file",
                arguments={"path": "out.txt", "content": "{@clip:missing.txt}"},
            )
        ],
    )

    expanded = expander.expand_response(response)

    assert expanded is response
    assert response.content == "note\n{@clip:missing-message.txt}"
    assert response.tool_calls[0].name == "write_file"
    assert isinstance(response.tool_calls[0].arguments, ClipboardExpansionErrorArgs)
    assert response.tool_calls[0].arguments.error.startswith(
        "Error expanding clipboard reference in tool arguments:"
    )


def test_expand_response_expands_message_when_available(tmp_path: Path) -> None:
    expander = ClipboardReferenceExpander(workspace=tmp_path)
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "msg.txt").write_text("expanded message", encoding="utf-8")

    response = LLMResponse(
        content="Before\n{@clip:notes/msg.txt}\nafter",
        tool_calls=[],
    )

    expanded = expander.expand_response(response)

    assert expanded is response
    assert response.content == "Before\nexpanded message\nafter"


def test_clipboard_reference_expander_blocks_outside_workspace_when_restricted(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}_blocked.txt"
    external.write_text("blocked", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside allowed directory"):
            ClipboardReferenceExpander(
                workspace=tmp_path,
                restrict_to_workspace=True,
            ).expand({"text": f"{{@clip:{external}}}"})
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
        arguments={"path": "out.txt", "content": "Before\n{@clip:snippets/snippet.txt}\nafter"},
    )
    calls = iter([
        LLMResponse(content="", tool_calls=[tool_call]),
        LLMResponse(content="Done", tool_calls=[]),
    ])
    loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(side_effect=lambda name, params: getattr(params, "error", None) or "ok")

    final_content, _, messages = await loop._run_agent_loop([])

    assert final_content == "Done"
    loop.tools.execute.assert_awaited_once_with(
        "write_file",
        {"path": "out.txt", "content": "Before\nexpanded text\nafter"},
    )
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"path": "out.txt", "content": "Before\\nexpanded text\\nafter"}'
    )
    assert messages[1]["content"] == "ok"


@pytest.mark.asyncio
async def test_run_agent_loop_adds_clipboard_expansion_failure_as_tool_result(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path)
    tool_call = ToolCallRequest(
        id="call1",
        name="write_file",
        arguments={"path": "out.txt", "content": "{@clip:missing.txt}"},
    )
    calls = iter([
        LLMResponse(content="", tool_calls=[tool_call]),
        LLMResponse(content="Done", tool_calls=[]),
    ])
    loop.provider.chat_with_retry = AsyncMock(side_effect=lambda *a, **kw: next(calls))
    loop.tools.get_definitions = MagicMock(return_value=[])
    loop.tools.execute = AsyncMock(side_effect=lambda name, params: getattr(params, "error", None) or "ok")

    final_content, tools_used, messages = await loop._run_agent_loop([])

    assert final_content == "Done"
    assert tools_used == ["write_file"]
    called_name, called_params = loop.tools.execute.await_args.args
    assert called_name == "write_file"
    assert isinstance(called_params, ClipboardExpansionErrorArgs)
    assert messages[0]["tool_calls"][0]["function"]["name"] == "write_file"
    assert messages[1]["content"].startswith(
        "Error expanding clipboard reference in tool arguments:"
    )


@pytest.mark.asyncio
async def test_run_agent_loop_expands_final_message_content(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "reply.txt").write_text("expanded reply", encoding="utf-8")

    loop = _make_loop(tmp_path)
    response = LLMResponse(content="Reply\n{@clip:notes/reply.txt}", tool_calls=[])
    loop.provider.chat_with_retry = AsyncMock(return_value=response)
    loop.tools.get_definitions = MagicMock(return_value=[])

    final_content, _, messages = await loop._run_agent_loop([])

    assert final_content == "Reply\nexpanded reply"
    assert messages[0]["content"] == "Reply\nexpanded reply"
