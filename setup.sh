#!/usr/bin/env bash
# Downloads the course framework into framework/, pinned to a fixed commit,
# links agent_code/rhine into it and applies the local framework patch.
# Safe to run repeatedly.
set -euo pipefail

UPSTREAM_URL="https://github.com/ukoethe/bomberman_rl.git"
UPSTREAM_COMMIT="0f55c1d207e586e6fe6db95e64302ed32281e370"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK="$ROOT/framework"
PATCH="$ROOT/patches/agents_logging.patch"
LINK="$FRAMEWORK/agent_code/rhine"

if [ ! -d "$FRAMEWORK/.git" ]; then
    git clone "$UPSTREAM_URL" "$FRAMEWORK"
fi
if ! git -C "$FRAMEWORK" cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null; then
    git -C "$FRAMEWORK" fetch origin
fi
git -C "$FRAMEWORK" checkout --quiet "$UPSTREAM_COMMIT"

if [ -L "$LINK" ]; then
    rm "$LINK"
elif [ -e "$LINK" ]; then
    echo "error: $LINK exists and is not a symlink" >&2
    exit 1
fi
ln -s ../../agent_code/rhine "$LINK"

if git -C "$FRAMEWORK" apply --check "$PATCH" 2>/dev/null; then
    git -C "$FRAMEWORK" apply "$PATCH"
    echo "Applied $(basename "$PATCH")."
elif git -C "$FRAMEWORK" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "$(basename "$PATCH") already applied."
else
    echo "error: $(basename "$PATCH") does not apply to framework/agents.py" >&2
    exit 1
fi

mkdir -p "$FRAMEWORK/logs"
echo "Framework ready in $FRAMEWORK"
