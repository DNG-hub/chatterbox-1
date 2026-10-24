#!/bin/sh
# Shim: the real guard lives in SDDvuePhone/scripts/agent-session.sh (one copy, no drift).
# Run from inside the repo. Everything the guard needs comes from the current git
# repo, so the canonical script behaves exactly as a local copy would. It is found
# next to this repo (E:/REPOS/SDDvuePhone on stable, ~/REPOS/SDDvuePhone on santos),
# or at $AGENT_SESSION_CANON if you set it.
here=$(cd "$(dirname "$0")" 2>/dev/null && pwd -P)/$(basename "$0")
common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || {
    echo "Not inside a git repository." >&2; exit 1; }
parent=$(dirname "$(dirname "$common")")
for c in "${AGENT_SESSION_CANON:-}" \
         "$parent/SDDvuePhone/scripts/agent-session.sh" \
         "$HOME/REPOS/SDDvuePhone/scripts/agent-session.sh" \
         "$HOME/repos/SDDvuePhone/scripts/agent-session.sh" \
         "/e/REPOS/SDDvuePhone/scripts/agent-session.sh"; do
    [ -n "$c" ] && [ -f "$c" ] || continue
    real=$(cd "$(dirname "$c")" && pwd -P)/$(basename "$c")
    [ "$real" = "$here" ] && continue          # never call ourselves
    exec sh "$c" "$@"
done
echo "agent-session.sh: cannot find the canonical script (SDDvuePhone/scripts/agent-session.sh)." >&2
echo "  Clone SDDvuePhone next to this repo, or set AGENT_SESSION_CANON to its full path." >&2
echo "  See E:/REPOS/MULTI-AGENT-RULES.md section 2." >&2
exit 1
