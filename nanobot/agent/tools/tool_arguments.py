"""Helpers for normalizing tool-call arguments before execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.agent.tools.filesystem import _resolve_path

_CLIP_REF_RE = re.compile(r"\{\{@clip:([^{}]+)\}\}")


@dataclass(frozen=True)
class ClipboardReferenceExpander:
    """Expand `{{@clip:...}}` placeholders using a fixed workspace policy."""

    workspace: Path
    restrict_to_workspace: bool = False

    def expand_tool_call(self, tool_call: Any) -> None:
        """Expand placeholders in a tool call's arguments in place."""
        tool_call.arguments = self.expand(tool_call.arguments)

    def expand(self, arguments: Any) -> Any:
        """Recursively replace placeholders with file contents using file-tool path rules."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        workspace_dir = self.workspace.resolve()

        def _read_clipboard_ref(raw_name: str) -> str:
            name = raw_name.strip()
            if not name:
                raise ValueError("empty clipboard filename")

            # `{{@clip:/foo.txt}}` means `workspace/foo.txt` when that file exists.
            candidate_name = name[1:] if name.startswith("/") else name
            workspace_candidate = (workspace_dir / candidate_name).resolve()
            try:
                workspace_candidate.relative_to(workspace_dir)
            except ValueError:
                workspace_candidate = None

            target_name = candidate_name if workspace_candidate and workspace_candidate.exists() else name

            try:
                path = _resolve_path(target_name, self.workspace, allowed_dir)
            except PermissionError as exc:
                raise ValueError(str(exc)) from exc
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"file not found: {target_name}")
            except OSError as exc:
                raise FileNotFoundError(f"file not found: {target_name}") from exc

            return path.read_text(encoding="utf-8", errors="replace")

        def _expand(value: Any) -> Any:
            if isinstance(value, str):
                return _CLIP_REF_RE.sub(lambda m: _read_clipboard_ref(m.group(1)), value)
            if isinstance(value, list):
                return [_expand(item) for item in value]
            if isinstance(value, dict):
                return {k: _expand(v) for k, v in value.items()}
            return value

        return _expand(arguments)


def expand_clipboard_references(
    arguments: Any,
    workspace: Path,
    *,
    restrict_to_workspace: bool = False,
) -> Any:
    """Backward-compatible wrapper around `ClipboardReferenceExpander`."""
    return ClipboardReferenceExpander(
        workspace=workspace,
        restrict_to_workspace=restrict_to_workspace,
    ).expand(arguments)
