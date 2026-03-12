"""Helpers for normalizing tool-call arguments before execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

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
        """Expand `{{@clip:...}}` placeholders once in each string value."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None

        def _read_clipboard_ref(raw_name: str) -> str:
            name = raw_name.strip()
            if not name:
                raise ValueError("empty clipboard filename")

            candidates = [name]
            if name.startswith("/"):
                candidates.insert(0, name[1:])

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
                return _CLIP_REF_RE.sub(lambda m: _read_clipboard_ref(m.group(1)), value)
            if isinstance(value, list):
                return [_expand(item) for item in value]
            if isinstance(value, dict):
                return {k: _expand(v) for k, v in value.items()}
            return value

        return _expand(arguments)
