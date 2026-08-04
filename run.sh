#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PYTHONDONTWRITEBYTECODE=1

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required. Install it from https://docs.astral.sh/uv/." >&2
    exit 127
fi

case "${1:-}" in
    locomo)
        CONFIG_PATH="config/locomo.toml"
        ;;
    longmemeval)
        CONFIG_PATH="config/longmemeval.toml"
        ;;
    *)
        echo "Usage: $0 {locomo|longmemeval}" >&2
        exit 2
        ;;
esac

uv run python src/main.py "$CONFIG_PATH"
