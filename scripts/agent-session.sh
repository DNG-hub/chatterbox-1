#!/bin/sh
#
# POSIX equivalent of agent-session.ps1, for agents that are not on Windows.
#
# Hermes runs on Santos (Linux), where ~/REPOS/* are sshfs mounts of stable's
# E:/REPOS/* — the same working tree and the same .git. PowerShell may not be
# present there, and the pre-commit guard only engages if a session recorded a
# baseline first, so without this Hermes would be permanently unguarded.
#
#   agent-session.sh start  "<task>" [agent-id]   check for duplicate work,
#                                                 record baseline, file lease
#   agent-session.sh claim  <path> [path...]      take ownership of a file
#   agent-session.sh status                       what is guarded right now
#   agent-session.sh scan                         read-only: every worktree,
#                                                 who owns it, is it reachable
#   agent-session.sh end                          close the session
#
# Add --force to `start` to proceed despite another agent's open lease on the
# same topic. The override is recorded in the lease.
#
# The baseline goes in the resolved git dir, so a worktree gets its own — which
# is what keeps two machines sharing one repo out of each other's way.
#
# Full doctrine: E:/REPOS/MULTI-AGENT-RULES.md
set -e

ACTION="${1:-status}"
GIT_DIR=$(git rev-parse --absolute-git-dir 2>/dev/null) || {
    echo "Not inside a git repository." >&2; exit 1; }
REPO_ROOT=$(git rev-parse --show-toplevel)
REPO_SLUG=$(basename "$REPO_ROOT" | tr '[:upper:]' '[:lower:]')
BASELINE="$GIT_DIR/agent-baseline.json"
VALESKA="${VALESKA_API_URL:-http://stable:8000}"

PYTHON=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PYTHON="$c"; break; }; done
[ -n "$PYTHON" ] || { echo "python3 is required." >&2; exit 1; }

# Shared lease reader/writer. Kept in one place because `start`, `status`,
# `scan` and `end` all need it and must agree about what "open" means.
#
# Leases live at an exact fact key, one per agent per repo:
#     repo.lease.<repo-slug>.<agent-id>
# Looking them up by address is the point. On 2026-08-09 a correct lease filed
# at 14:37Z was invisible to a similarity search three hours later, and the
# recent-thoughts window is consumed by auto-capture noise in about twenty
# minutes. Neither can answer "is anyone already doing this?" - an exact key can.
KNOWN_AGENTS="claude-code codex hermes-exo openclaw"

lease_py() {
    VALESKA="$VALESKA" SLUG="$REPO_SLUG" AGENTS="$KNOWN_AGENTS" "$PYTHON" - "$@" <<'PY'
import json, os, sys, urllib.request, urllib.error

BASE = os.environ["VALESKA"].rstrip("/")
SLUG = os.environ["SLUG"]

def _req(url, data=None, agent="claude-code"):
    r = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    r.add_header("X-Valeska-Agent-Id", agent)
    if data:
        r.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(r, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))

def open_leases(exclude):
    """Every OTHER agent's open lease. Exact-key lookups first, then a shallow
    scrape for agents still filing leases the old way."""
    out, closed, reached = {}, set(), False
    for a in os.environ["AGENTS"].split():
        if a == exclude:
            continue
        try:
            f = _req("%s/api/v1/memory/facts/repo.lease.%s.%s?project=%s" % (BASE, SLUG, a, SLUG))
            reached = True
            v = json.loads(f.get("fact_value") or "{}")
            if str(v.get("status")) == "closed":
                # A closed key must also silence the older "REPO LEASE (active)"
                # thought this agent filed on start, or every finished agent
                # stays open forever and the warning becomes noise.
                closed.add(a)
                continue
            out[a] = {"agent_id": a, "started_at": v.get("started_at", ""), "task": v.get("task", "")}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                reached = True          # no lease for that agent, which is an answer
        except Exception:
            pass
    try:
        # thought_type=memory keeps per-prompt and per-tool capture noise from
        # eating the whole 100-item window.
        r = _req("%s/api/v1/memory/thoughts/recent?project=%s&thought_type=memory&limit=100" % (BASE, SLUG))
        reached = True
        for t in r.get("thoughts", []):
            tags = t.get("tags") or []
            owner = t.get("agent_id")
            if "repo-lease" not in tags or not owner or owner == exclude:
                continue
            if owner in out or owner in closed:   # the keyed fact wins
                continue
            if "closed" in tags or "status:closed" in tags:
                continue
            out[owner] = {"agent_id": owner, "started_at": t.get("created_at", ""),
                          "task": summarise(t.get("content", ""))}
    except Exception:
        pass
    return reached, list(out.values())

def summarise(content):
    """Leases arrive either as a sentence or as `key = {json}`. Print the task,
    not the schema URL."""
    first = (content or "").split("\n")[0].strip()
    b = first.find("{")
    if b >= 0:
        try:
            o = json.loads(first[b:])
            for k in ("task_summary", "task", "summary"):
                if o.get(k):
                    return str(o[k])
        except Exception:
            pass
        first = first[:b].rstrip(" =")
    return first[:140]

STOP = set("the a an and or of for to in on with from into at by is are be was were this that "
           "it its as plan work task update fix add make new all some repo project file files".split())

def topic_words(text):
    w = "".join(c if c.isalnum() else " " for c in (text or "")).lower().split()
    return {x for x in w if len(x) > 3 and x not in STOP}

def write_lease(agent, value):
    body = json.dumps({
        "fact_key": "repo.lease.%s.%s" % (SLUG, agent),
        "fact_value": json.dumps(value),
        "project": SLUG, "agent_id": agent,
        "tags": ["repo-lease", "repo:" + SLUG, "agent:" + agent, "status:" + value.get("status", "active")],
        "visibility_scope": "shared_team",
    }).encode()
    try:
        _req("%s/api/v1/memory/facts" % BASE, body, agent)
        return True
    except Exception:
        return False

cmd = sys.argv[1]

if cmd == "check":                       # check <agent> <task>
    agent, task = sys.argv[2], sys.argv[3]
    reached, leases = open_leases(agent)
    if not reached:
        print("  (Valeska unreachable - CANNOT check for duplicate work)")
        print("   Check by hand before starting, or you may repeat someone.")
        sys.exit(0)
    if not leases:
        sys.exit(0)
    mine = topic_words(task)
    clash = []
    print("")
    print("  %d other agent lease(s) OPEN on %s:" % (len(leases), SLUG))
    for l in leases:
        shared = mine & topic_words(l["task"])
        mark = "  <-- OVERLAPS YOUR TASK" if len(shared) >= 2 else ""
        print("      %s  %s  %s%s" % ((l["started_at"] or "")[:16], l["agent_id"], l["task"][:90], mark))
        if len(shared) >= 2:
            clash.append((l, sorted(shared)))
    print("  Read these before you start. Duplication is judged on topic, not paths.")
    if clash:
        print("")
        print("  STOPPED - another agent is already on this topic.")
        for l, shared in clash:
            print("      %s: shared terms [%s]" % (l["agent_id"], ", ".join(shared)))
        print("")
        print("  Do one of these instead of restarting their work:")
        print("    1. Read their artifact and extend it; say so in your task sentence.")
        print("    2. Ask Dave which of you owns it.")
        print("    3. Narrow your task so it genuinely does not overlap, and reword it.")
        print("")
        print("  If you are certain it is separate work, re-run with --force.")
        print("  See E:/REPOS/MULTI-AGENT-RULES.md section 3.")
        sys.exit(2)
    sys.exit(0)

elif cmd == "list":                      # list <agent>
    _, leases = open_leases(sys.argv[2])
    if not leases:
        print("  none")
    for l in leases:
        print("  %s  %s  %s" % ((l["started_at"] or "")[:16], l["agent_id"], l["task"][:100]))

elif cmd == "open":                      # open <agent> <json-value>
    ok = write_lease(sys.argv[2], json.loads(sys.argv[3]))
    print("  lease filed to Valeska  ->  repo.lease.%s.%s" % (SLUG, sys.argv[2]) if ok
          else "  keyed lease FAILED - other agents may not see this lease.")

elif cmd == "close":                     # close <agent> <task>
    write_lease(sys.argv[2], {"status": "closed", "task": sys.argv[3], "agent_id": sys.argv[2]})
PY
}

case "$ACTION" in

start)
    TASK="${2:-}"
    AGENT="${3:-hermes-exo}"
    FORCE=0
    for a in "$@"; do [ "$a" = "--force" ] && FORCE=1; done
    [ "$AGENT" = "--force" ] && AGENT="hermes-exo"

    # Duplicate check BEFORE anything is written or any worktree is taken.
    # Isolation is what makes two agents invisible to each other, so the moment
    # to look is before you take any.
    set +e
    lease_py check "$AGENT" "$TASK"
    CHECK=$?
    set -e
    if [ "$CHECK" = "2" ] && [ "$FORCE" = "0" ]; then
        exit 2
    fi
    if [ "$CHECK" = "2" ]; then
        echo ""
        echo "  --force: proceeding despite an overlapping lease. Recorded."
    fi

    git status --porcelain | AGENT="$AGENT" TASK="$TASK" BASELINE="$BASELINE" "$PYTHON" -c '
import json, os, subprocess, sys, datetime

dirty = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if len(line) < 4:
        continue
    p = line[3:].strip().strip(chr(34))
    if " -> " in p:
        p = p.split(" -> ")[-1].strip().strip(chr(34))
    dirty.append(p.replace(chr(92), "/"))

def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()

baseline = {
    "agent_id": os.environ["AGENT"],
    "task": os.environ["TASK"],
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "base_commit": git("rev-parse", "HEAD"),
    "started_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dirty_at_start": dirty,
    "claimed": [],
}
# No BOM: the checker reads utf-8-sig now, but writing one would still be wrong.
with open(os.environ["BASELINE"], "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2)

print("")
print("Session started: %s on %s" % (baseline["agent_id"], baseline["branch"]))
print("  base commit : %s" % baseline["base_commit"])
if baseline["task"]:
    print("  task        : %s" % baseline["task"])
if dirty:
    print("")
    print("  %d file(s) were ALREADY modified before you started:" % len(dirty))
    for p in dirty[:12]:
        print("      %s" % p)
    if len(dirty) > 12:
        print("      ... and %d more" % (len(dirty) - 12))
    print("")
    print("  Someone else is probably mid-task. You will be blocked from")
    print("  committing these unless you claim them. Consider working in your")
    print("  own worktree instead:")
    print("      git worktree add -b agent/%s ../.agent-worktrees/%s-%s"
          % (os.environ["AGENT"], os.path.basename(os.getcwd()), os.environ["AGENT"]))
else:
    print("  working tree clean - nothing to collide with")
'
    # Best-effort lease. Valeska being unreachable must never stop work.
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    HEAD_SHA=$(git rev-parse HEAD)
    BODY=$(AGENT="$AGENT" TASK="$TASK" SLUG="$REPO_SLUG" BRANCH="$BRANCH" SHA="$HEAD_SHA" \
        "$PYTHON" -c '
import json, os
print(json.dumps({
    "content": "REPO LEASE (active): %s on %s/%s - %s" % (
        os.environ["AGENT"], os.environ["SLUG"], os.environ["BRANCH"], os.environ["TASK"]),
    "thought_type": "memory",
    "project": os.environ["SLUG"],
    "agent_id": os.environ["AGENT"],
    "tags": ["repo-lease", "repo:" + os.environ["SLUG"], "agent:" + os.environ["AGENT"], "active"],
    "visibility_scope": "shared_team",
}))')
    if curl -s -m 10 -o /dev/null -X POST "$VALESKA/api/v1/memory/thoughts" \
        -H "Content-Type: application/json" -H "X-Valeska-Agent-Id: $AGENT" -d "$BODY" 2>/dev/null; then
        echo ""
    else
        echo ""
        echo "  (Valeska unreachable - lease not filed, local guard still active)"
    fi
    # The durable half: an exact key another agent's `start` can find later.
    LEASE_VALUE=$(AGENT="$AGENT" TASK="$TASK" BRANCH="$BRANCH" SHA="$HEAD_SHA" FORCED="$FORCE" "$PYTHON" -c '
import json, os, datetime
print(json.dumps({
    "status": "active",
    "task": os.environ["TASK"],
    "agent_id": os.environ["AGENT"],
    "branch": os.environ["BRANCH"],
    "worktree": os.getcwd(),
    "base_commit": os.environ["SHA"],
    "started_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "forced_over_open_lease": os.environ["FORCED"] == "1",
}))')
    lease_py open "$AGENT" "$LEASE_VALUE"
    echo ""
    ;;

claim)
    shift
    [ $# -gt 0 ] || { echo "Nothing to claim. Pass one or more paths." >&2; exit 1; }
    [ -f "$BASELINE" ] || { echo "No session. Run: agent-session.sh start \"<task>\"" >&2; exit 1; }
    BASELINE="$BASELINE" "$PYTHON" - "$@" <<'PY'
import json, os, sys
path = os.environ["BASELINE"]
with open(path, encoding="utf-8-sig") as f:
    b = json.load(f)
b["claimed"] = sorted(set(b.get("claimed", [])) | {a.replace("\\", "/") for a in sys.argv[1:]})
with open(path, "w", encoding="utf-8") as f:
    json.dump(b, f, indent=2)
print("Claimed %d path(s); they will no longer block a commit:" % len(sys.argv[1:]))
for a in sys.argv[1:]:
    print("    %s" % a)
PY
    ;;

status)
    if [ -f "$BASELINE" ]; then
        BASELINE="$BASELINE" "$PYTHON" -c '
import json, os
with open(os.environ["BASELINE"], encoding="utf-8-sig") as f:
    b = json.load(f)
guarded = [p for p in b.get("dirty_at_start", []) if p not in b.get("claimed", [])]
print("")
print("Session: %s since %s" % (b.get("agent_id"), b.get("started_at")))
print("  branch   : %s" % b.get("branch"))
print("  guarded  : %d file(s) treated as someone else%ss" % (len(guarded), chr(39)))
for p in guarded[:10]:
    print("      %s" % p)
print("  claimed  : %d" % len(b.get("claimed", [])))
print("")'
    else
        echo ""
        echo "No session recorded. The pre-commit guard is INACTIVE."
        echo "Start one with: agent-session.sh start \"<task>\" hermes-exo"
        echo ""
    fi
    echo "Other agents with open leases:"
    lease_py list "${2:-}"
    echo ""
    ;;

scan)
    # Read-only. Answers "is anyone else in here?" without touching anything.
    #
    # git marks a worktree prunable whenever its folder is unreachable from THIS
    # machine. Worktrees made on santos live on santos's local disk while their
    # bookkeeping syncs through the shared .git, so stable sees live work as
    # stale. On 2026-08-09 that nearly cost ~2,000 lines of uncommitted work.
    echo ""
    echo "Worktrees registered for $REPO_SLUG:"
    UNREACHABLE=0
    while IFS= read -r line; do
        case "$line" in
            worktree\ *)
                WT="${line#worktree }"
                if [ -d "$WT" ]; then
                    echo "  [reachable  ] $WT"
                else
                    echo "  [NOT ON THIS MACHINE] $WT"
                    UNREACHABLE=1
                    echo "$WT" >> /tmp/.agent-scan-unreachable.$$
                fi
                ;;
            branch\ *) echo "                  branch: ${line#branch refs/heads/}" ;;
        esac
    done <<EOF
$(git worktree list --porcelain 2>/dev/null)
EOF
    if [ "$UNREACHABLE" = "1" ]; then
        echo ""
        echo "  DO NOT PRUNE. Unreachable does not mean abandoned."
        echo "  Those paths are very likely alive on another machine. To check:"
        while IFS= read -r p; do
            case "$p" in
                /home/*) echo "      ls -la $p        (santos-local; from stable: ssh dave@santos)" ;;
                *)       echo "      inspect: $p" ;;
            esac
        done < /tmp/.agent-scan-unreachable.$$
        echo "  Then search Valeska for a lease from the owning agent."
        echo "  Only Dave may authorise 'git worktree prune' or 'remove'."
    fi
    rm -f /tmp/.agent-scan-unreachable.$$
    echo ""
    echo "Other agents with open leases:"
    lease_py list "${2:-}"
    echo ""
    ;;

end)
    AGENT="${2:-hermes-exo}"
    TASK_DONE=""
    if [ -f "$BASELINE" ]; then
        TASK_DONE=$(BASELINE="$BASELINE" "$PYTHON" -c '
import json, os
with open(os.environ["BASELINE"], encoding="utf-8-sig") as f:
    print(json.load(f).get("task", ""))')
    fi
    # Close the keyed fact too, or the next agent to run `start` is stopped by a
    # lease nobody is holding any more.
    lease_py close "$AGENT" "$TASK_DONE"
    rm -f "$BASELINE"
    echo "Session closed. Guard removed; nothing was committed or cleaned."
    echo ""
    echo "Leave a receipt: commit, branch, stash, patch, or a note naming the"
    echo "exact paths left dirty. Never report done on unexplained dirty state."
    ;;

*)
    echo "Usage: agent-session.sh {start \"<task>\" [agent-id] [--force]|claim <path>...|status|scan|end [agent-id]}" >&2
    exit 1
    ;;
esac
