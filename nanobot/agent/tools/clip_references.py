"""Expand clip placeholders in assistant/tool text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.filesystem import _resolve_path

_CLIP_REF_LINE_RE = re.compile(
    r"(?m)^[ \t]*\{[ \t]*@clip[ \t]*:[ \t]*([^{}\n]+?)[ \t]*\}[ \t]*$"
)


class ClipboardExpansionErrorArgs(dict):
    """Dictionary-like tool arguments carrying a precomputed expansion error."""

    def __init__(self, original_arguments: dict[str, Any], error: str):
        super().__init__(original_arguments)
        self.error = error


@dataclass(frozen=True)
class ClipboardReferenceExpander:
    """Expand standalone `{@clip:...}` lines using workspace-aware path rules."""

    workspace: Path
    restrict_to_workspace: bool = False
    _TOOL_CALL_ERROR_PREFIX = "Error expanding clipboard reference in tool arguments: "

    def expand_tool_call(self, tool_call: Any) -> str | None:
        """Expand placeholders in a tool call's arguments in place.

        Returns an error string when expansion fails so callers can attach it
        as a tool result without duplicating exception handling.
        """
        try:
            tool_call.arguments = self.expand(tool_call.arguments)
        except (FileNotFoundError, OSError, ValueError) as e:
            error = f"{self._TOOL_CALL_ERROR_PREFIX}{e}"
            logger.warning("Tool call argument expansion failed for {}: {}", tool_call.name, error)
            return error
        return None

    def expand_message(self, message: str | None) -> str | None:
        """Expand clipboard references in assistant message content.

        If expansion fails, keep the original message unchanged.
        """
        if not message:
            return message
        try:
            expanded = self.expand(message)
        except (FileNotFoundError, OSError, ValueError):
            return message
        return expanded if isinstance(expanded, str) else message

    def expand_response(self, response: Any) -> Any:
        """Expand clipboard references in a response's message and tool calls."""
        if hasattr(response, "content"):
            response.content = self.expand_message(response.content)

        for tool_call in getattr(response, "tool_calls", []):
            if error := self.expand_tool_call(tool_call):
                tool_call.arguments = ClipboardExpansionErrorArgs(tool_call.arguments, error)
        return response

    def expand(self, arguments: Any) -> Any:
        """Expand standalone `{@clip:...}` lines in each string value."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None

        def _read_clipboard_ref(raw_name: str) -> str:
            name = raw_name.strip()
            if not name:
                raise ValueError("empty clipboard filename")

            candidates: list[str] = []

            def _add_candidate(candidate: str) -> None:
                if candidate and candidate not in candidates:
                    candidates.append(candidate)

            relative_name = name[1:] if name.startswith("/") else name
            _add_candidate(relative_name)
            if relative_name.startswith("work_dir/"):
                _add_candidate(relative_name[len("work_dir/"):])
            if name.startswith("/"):
                _add_candidate(name)

            permission_error: PermissionError | None = None
            for candidate in candidates:
                try:
                    path = _resolve_path(candidate, self.workspace, allowed_dir)
                except PermissionError as exc:
                    permission_error = exc
                    continue
                try:
                    if not path.is_file():
                        continue
                except OSError:
                    continue

                logger.info("Expanding clipboard reference {} -> {}", name, path)
                return path.read_text(encoding="utf-8", errors="replace")

            if permission_error is not None:
                raise ValueError(str(permission_error)) from permission_error
            raise FileNotFoundError(f"file not found: {name}")

        def _expand(value: Any) -> Any:
            if isinstance(value, str):
                return _CLIP_REF_LINE_RE.sub(lambda m: _read_clipboard_ref(m.group(1)), value)
            if isinstance(value, list):
                return [_expand(item) for item in value]
            if isinstance(value, dict):
                return {k: _expand(v) for k, v in value.items()}
            return value

        return _expand(arguments)
