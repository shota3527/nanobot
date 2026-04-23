#!/bin/sh
set -eu

home="${HOME:-/home/nanobot}"
config_dir="$home/.nanobot"
base_config="$config_dir/config.json"
hardened_config="${NANOBOT_HARDENED_CONFIG:-/tmp/nanobot-hardened-config.json}"

if [ -d "$config_dir" ] && [ ! -w "$config_dir" ]; then
    owner_uid=$(stat -c %u "$config_dir" 2>/dev/null || stat -f %u "$config_dir" 2>/dev/null)
    cat >&2 <<EOF
Error: $config_dir is not writable (owned by UID $owner_uid, running as UID $(id -u)).

Fix (pick one):
  Host:   sudo chown -R 1000:1000 ~/.nanobot
  Docker: docker run --user \$(id -u):\$(id -g) ...
  Podman: podman run --userns=keep-id ...
EOF
    exit 1
fi

python - "$base_config" "$hardened_config" <<'PY'
import json
import os
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
data = {}

if src.exists():
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error: failed to read {src}: {exc}", file=sys.stderr)
        sys.exit(1)

tools = data.setdefault("tools", {})
exec_cfg = tools.setdefault("exec", {})

exec_cfg["enable"] = False
exec_cfg["sandbox"] = ""
tools["restrictToWorkspace"] = True

dst.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

case "${1:-}" in
    gateway|serve|agent)
        for arg in "$@"; do
            if [ "$arg" = "--config" ] || [ "$arg" = "-c" ]; then
                exec nanobot "$@"
            fi
        done
        exec nanobot "$@" --config "$hardened_config"
        ;;
esac

exec nanobot "$@"
