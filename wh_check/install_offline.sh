#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELS_DIR="${SCRIPT_DIR}/wheelhouse"
REQ_FILE="${SCRIPT_DIR}/requirements.lock"

if [ ! -d "$WHEELS_DIR" ]; then
    echo "❌ Error: Wheelhouse directory not found at $WHEELS_DIR"
    exit 1
fi

echo "⚡ Fast-installing offline ML stack using uv..."

# Ensure uv is available (installs from wheelhouse if needed)
if ! command -v uv &> /dev/null; then
    echo "Installing uv from wheelhouse..."
    python3 -m pip install -q --no-index --find-links="$WHEELS_DIR" uv
fi

# Ultra-fast installation into environment
uv pip install --system --no-index --find-links="$WHEELS_DIR" -r "$REQ_FILE"
echo "✅ Offline installation complete!"
