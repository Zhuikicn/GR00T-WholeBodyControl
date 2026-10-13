#!/usr/bin/env bash
# Install the standalone second-view renderer into this repository's environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RENDER_ENV="$REPO_ROOT/.venv_second_view"

if [[ ! -x "$RENDER_ENV/bin/python" ]]; then
    uv venv "$RENDER_ENV" --python 3.10 --prompt sonic_second_view
fi
uv pip install --python "$RENDER_ENV/bin/python" "torch==2.11.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$RENDER_ENV/bin/python" -e "$REPO_ROOT/gear_sonic[second_view]"

echo "Second-view renderer installed in $RENDER_ENV"
echo "Run from $REPO_ROOT:"
echo "  .venv_second_view/bin/python -m gear_sonic.scripts.run_second_view_renderer --state-host <SONIC-IP>"
