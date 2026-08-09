<#
.SYNOPSIS
    Declare what an agent is working on, so several can share a repository.

.DESCRIPTION
    Claude Code, Codex and Hermes all work these repositories, sometimes at the
    same time and sometimes in the same working tree. On 2026-08-08 two of them
    did exactly that: one ran `git add -A backend` and swept the other's
    half-finished migration into an unrelated commit, and a container restart
    applied that migration before its author intended.

    A lease schema already existed in Valeska as `repo.lease.template.v1`. In the
    hours after it was written, no agent filed a single lease against it  -
    including the one that caused the collision. A convention that requires
    hand-assembling JSON and remembering to do so is a convention nobody follows.
    This makes it one command, and pairs it with a pre-commit hook that notices
    when it has not been run.

    A second incident on 2026-08-09 showed filing leases was never the gap.
    claude-code filed a correct, discoverable lease at 14:37Z; hermes-exo began
    the same work at 17:25Z and produced a parallel plan. Nothing made anybody
    look, and the two agents' file scopes never overlapped, so no path-based
    check could have caught it. `start` now runs the check itself and stops on a
    topic clash.

    start    Check for duplicate work, record what is dirty, file a lease.
    claim    Take ownership of files that were dirty before you started.
    status   Show the baseline, the claim list, and who else is active.
    scan     Read-only: every worktree, who owns it, whether it is reachable.
    end      Close the lease. Does not commit, push, or clean anything.

    Full doctrine: E:\REPOS\MULTI-AGENT-RULES.md

.PARAMETER Action
    start | claim | status | scan | end

.PARAMETER Force
    Start anyway when another agent holds an open lease on the same topic. The
    override is recorded in the lease.

.PARAMETER Task
    One sentence on what this session is for. Used by `start`.

.PARAMETER Paths
    Comma-separated paths or globs. Used by `start` (exclusive_paths) and
    `claim`.

.PARAMETER AgentId
    Who you are: claude-code, codex, hermes-exo. Defaults to claude-code.

.EXAMPLE
    scripts\agent-session.ps1 start -Task "discount schedule" -Paths "backend/app/services/**"

.EXAMPLE
    scripts\agent-session.ps1 claim -Paths "SDD/index.md"

.EXAMPLE
    scripts\agent-session.ps1 status
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('start', 'claim', 'status', 'end', 'scan')]
    [string]$Action,

    [string]$Task = '',
    [string]$Paths = '',
    [string]$AgentId = 'claude-code',
    [string]$ValeskaUrl = 'http://127.0.0.1:8000',

    # Proceed even though another agent holds an active lease on the same topic.
    # Recorded in the lease, because it is a decision you have to be able to
    # defend - see E:\REPOS\MULTI-AGENT-RULES.md section 3.
    [switch]$Force,

    # Isolation. A worktree is created automatically when the tree is ALREADY
    # dirty, because that is precisely the case where two agents collide - and
    # not when it is clean, because isolating a lone agent is pure friction.
    # -Worktree forces one; -NoWorktree declines even on a dirty tree.
    [switch]$Worktree,
    [switch]$NoWorktree
)

$ErrorActionPreference = 'Stop'

$repoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $repoRoot) { Write-Host 'Not inside a git repository.' -ForegroundColor Red; exit 1 }
$repoRoot = $repoRoot -replace '/', '\'
$repoSlug = (Split-Path $repoRoot -Leaf).ToLower()
# Resolved, not assumed. Inside a worktree, .git is a FILE pointing at
# .git/worktrees/<name>, so a hard-coded '.git\...' path writes a baseline the
# hook will never find - switching the guard off silently in exactly the
# isolated checkouts most likely to be shared. Resolving it also gives every
# worktree its own baseline, which is how two agents, or two machines sharing
# one repo over sshfs, stay out of each other's way.
$gitDir = (git rev-parse --absolute-git-dir 2>$null)
if (-not $gitDir) { $gitDir = Join-Path $repoRoot '.git' }
$baselinePath = Join-Path ($gitDir -replace '/', '\') 'agent-baseline.json'

# Every agent that works these repos. Leases are stored as a keyed fact per
# agent, so this list is what makes a lease lookup exact rather than a search.
# Adding a new agent means adding it here and in scripts/agent-session.sh.
$knownAgents = @('claude-code', 'codex', 'hermes-exo', 'openclaw')
$leaseKey = "repo.lease.$repoSlug.$AgentId"

function Write-LeaseFact {
    # The durable half of a lease. Same key every time for this agent in this
    # repo, so `start` can find it later by address instead of hoping a
    # similarity search ranks it highly enough to appear.
    param([hashtable]$Value)
    try {
        $body = @{
            fact_key   = $leaseKey
            fact_value = ($Value | ConvertTo-Json -Depth 6 -Compress)
            project    = $repoSlug
            agent_id   = $AgentId
            tags       = @('repo-lease', "repo:$repoSlug", "agent:$AgentId", "status:$($Value.status)")
            visibility_scope = 'shared_team'
        } | ConvertTo-Json -Depth 8
        $null = Invoke-RestMethod -Uri "$ValeskaUrl/api/v1/memory/facts" -Method Post `
            -ContentType 'application/json' -TimeoutSec 20 `
            -Headers @{ 'X-Valeska-Agent-Id' = $AgentId } -Body $body
        return $true
    } catch { return $false }
}

function Get-DirtyPaths {
    # Porcelain v1: two status chars, a space, then the path. Renames read
    # "old -> new"; the new name is what a later commit would stage.
    $out = @()
    foreach ($line in (git status --porcelain 2>$null)) {
        if ($line.Length -lt 4) { continue }
        $path = $line.Substring(3).Trim().Trim('"')
        if ($path -match '\s->\s') { $path = ($path -split '\s->\s')[-1].Trim().Trim('"') }
        $out += ($path -replace '\\', '/')
    }
    return $out
}

function New-AgentWorktree {
    param([string]$RepoRoot, [string]$Slug)

    # Outside the repository on purpose. A worktree nested inside the main tree
    # shows up as untracked files in every `git status` the other agent runs,
    # which is the confusion this is meant to remove.
    $base = Join-Path (Split-Path $RepoRoot -Parent) '.agent-worktrees'
    if (-not (Test-Path $base)) { New-Item -ItemType Directory -Path $base -Force | Out-Null }

    $repoName = Split-Path $RepoRoot -Leaf
    $path = Join-Path $base "$repoName-$Slug"
    $branch = "agent/$Slug"

    if (Test-Path $path) {
        Write-Host "  worktree already exists: $path" -ForegroundColor Yellow
        return $path
    }

    # Branch from the current HEAD so work in progress on this branch is present.
    $out = git -C $RepoRoot worktree add -b $branch $path 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host '  could not create a worktree:' -ForegroundColor Yellow
        Write-Host "      $out"
        Write-Host '  continuing in the shared tree - the commit guard still applies.'
        return $null
    }
    return $path
}

function Get-ActiveLeases {
    # Every OTHER agent's open lease on this repo. Returned, not printed, so
    # both `start` and `status` show the same thing.
    #
    # Filed leases were never the problem. On 2026-08-09 claude-code filed a
    # correct lease at 14:37Z and hermes-exo started the same work at 17:25Z
    # anyway, because nothing made anybody look. `start` calls this now.
    param([string]$ExcludeAgent)

    # TWO sources on purpose, because neither alone is sufficient:
    #
    #  * /thoughts/recent is deterministic but shallow. The auto-capture hooks
    #    fire on every prompt and tool call, so on a busy repo 100 thoughts can
    #    be twenty minutes. Anything older falls out of the window.
    #  * /thoughts/search reaches back further but ranks by similarity, and
    #    ranking is not answering. Asked for "REPO LEASE active storyteller" it
    #    returned ZERO of the three leases that existed - which is precisely how
    #    a duplicate-work check quietly reports "all clear".
    #
    # Merged, deduplicated by thought id. `limit` above 100 returns an empty
    # list rather than an error, so it is capped here.
    $found = @{}
    $closedAgents = @{}
    $reached = $false

    # SOURCE 1, and the one to trust: a keyed fact per agent, at the address
    # every agent is told to use in E:\REPOS\MULTI-AGENT-RULES.md:
    #     repo.lease.<repo-slug>.<agent-id>
    # Facts are looked up by exact key, so this has no recency window at all - a
    # lease filed six hours ago is as findable as one filed a minute ago. That
    # is the whole reason it exists: the thought stores below could not see a
    # lease from earlier the same afternoon.
    foreach ($other in $knownAgents) {
        if ($ExcludeAgent -and $other -eq $ExcludeAgent) { continue }
        try {
            $f = Invoke-RestMethod -Method Get -TimeoutSec 10 `
                -Uri "$ValeskaUrl/api/v1/memory/facts/repo.lease.$repoSlug.$other`?project=$repoSlug" `
                -Headers @{ 'X-Valeska-Agent-Id' = $AgentId }
            $reached = $true
            if (-not $f.fact_value) { continue }
            $v = $f.fact_value | ConvertFrom-Json
            # A closed key must also silence the older "REPO LEASE (active)"
            # thought this agent filed when it started, or every finished agent
            # stays open forever and the check becomes noise people learn to
            # ignore.
            if ("$($v.status)" -eq 'closed') { $closedAgents[$other] = $true; continue }
            $found[$other] = [pscustomobject]@{
                agent_id = $other; created_at = $v.started_at
                summary  = "$($v.task)"; closed = $false
            }
        } catch {
            # 404 just means that agent holds no lease here.
            if ($_.Exception.Response.StatusCode.value__ -eq 404) { $reached = $true }
        }
    }

    # SOURCE 2: recent thoughts, for agents filing leases the older way or with
    # their own task-scoped keys. thought_type=memory keeps the per-prompt and
    # per-tool auto-capture noise out of the window, which otherwise consumes
    # the whole 100-item limit in about twenty minutes.
    $raw = @()
    try {
        $r = Invoke-RestMethod -Method Get -TimeoutSec 20 `
            -Uri "$ValeskaUrl/api/v1/memory/thoughts/recent?project=$repoSlug&thought_type=memory&limit=100" `
            -Headers @{ 'X-Valeska-Agent-Id' = $AgentId }
        $raw += @($r.thoughts); $reached = $true
    } catch { }
    foreach ($q in @("REPO LEASE active $repoSlug", "repo lease $repoSlug agent working")) {
        try {
            $body = @{ query = $q; project = $repoSlug; limit = 50 } | ConvertTo-Json
            $r = Invoke-RestMethod -Uri "$ValeskaUrl/api/v1/memory/thoughts/search" -Method Post `
                -ContentType 'application/json' -TimeoutSec 20 `
                -Headers @{ 'X-Valeska-Agent-Id' = $AgentId } -Body $body
            $raw += @($r.thoughts); $reached = $true
        } catch { }
    }
    if (-not $reached) {
        Write-Host '  (Valeska unreachable - CANNOT check for duplicate work)' -ForegroundColor Yellow
        Write-Host '   Check by hand before starting, or you may repeat someone.' -ForegroundColor Yellow
        return @()
    }

    # Latest lease record per agent wins. Agents use different status
    # vocabularies - the scripts write bare 'active'/'closed', hermes-exo writes
    # 'status:active', 'status:closed', 'status:completed-review-pending' - so
    # this treats a lease as OPEN unless it is explicitly closed. Over-warning
    # costs a glance; under-warning costs a duplicated day.
    $seen = @{}; $byAgent = $found
    foreach ($t in $raw) {
        if (-not $t) { continue }
        if ($seen.ContainsKey($t.thought_id)) { continue }
        $seen[$t.thought_id] = $true
        $tags = @($t.tags)
        if ($tags -notcontains 'repo-lease') { continue }
        if ($tags -notcontains "repo:$repoSlug") { continue }
        $owner = $t.agent_id
        if (-not $owner) { continue }
        if ($ExcludeAgent -and $owner -eq $ExcludeAgent) { continue }
        # The keyed fact is authoritative. Only fall back to a scraped thought
        # for an agent that has not filed one - and never compare the two
        # timestamps, which arrive as a string from one source and a [DateTime]
        # from the other.
        if ($found.ContainsKey($owner)) { continue }
        if ($closedAgents.ContainsKey($owner)) { continue }
        if (-not $byAgent.ContainsKey($owner) -or $byAgent[$owner].created_at -lt $t.created_at) {
            $byAgent[$owner] = [pscustomobject]@{
                agent_id   = $owner
                created_at = $t.created_at
                summary    = (Get-LeaseSummary $t.content)
                closed     = (($tags -contains 'closed') -or ($tags -contains 'status:closed'))
            }
        }
    }
    return @($byAgent.Values | Where-Object { -not $_.closed } | Sort-Object created_at -Descending)
}

function Get-LeaseSummary {
    # Leases arrive in two shapes. The scripts write a sentence:
    #   "REPO LEASE (active): claude-code on storyteller/main - <task>"
    # hermes-exo writes a fact line with JSON after an equals sign:
    #   "repo.lease.storyteller.t_6d0cad1b = {"task_summary": "...", ...}"
    # Printing the raw second form fills the screen with a schema URL and hides
    # the one sentence that decides whether you are about to repeat someone.
    param([string]$Content)
    $first = (($Content -split "`n")[0]).Trim()
    $brace = $first.IndexOf('{')
    if ($brace -ge 0) {
        try {
            $obj = $first.Substring($brace) | ConvertFrom-Json
            foreach ($f in @('task_summary', 'task', 'summary')) {
                if ($obj.PSObject.Properties.Name -contains $f -and $obj.$f) {
                    return "$($obj.$f)"
                }
            }
        } catch { }
        $first = $first.Substring(0, $brace).TrimEnd(' ', '=')
    }
    if ($first.Length -gt 140) { $first = $first.Substring(0, 140) + '...' }
    return $first
}

function Format-LeaseTime {
    # Invoke-RestMethod deserialises ISO timestamps into [DateTime], but the
    # search and recent endpoints do not always agree, so handle both rather
    # than calling .Substring() on something that is not a string.
    param($Value)
    if ($null -eq $Value) { return '(undated)' }
    if ($Value -is [datetime]) { return $Value.ToUniversalTime().ToString('yyyy-MM-dd HH:mm') }
    $s = [string]$Value
    if ($s.Length -ge 16) { return $s.Substring(0, 16).Replace('T', ' ') }
    return $s
}

function Get-TopicWords {
    # Words that carry meaning. Everything else is noise that would match every
    # lease against every other lease.
    param([string]$Text)
    $stop = @('the','a','an','and','or','of','for','to','in','on','with','from','into','at','by',
              'is','are','be','was','were','this','that','it','its','as','plan','work','task',
              'update','fix','add','make','new','all','some','repo','project','file','files')
    return @(($Text -replace '[^a-zA-Z0-9 ]', ' ') -split '\s+' |
        ForEach-Object { $_.ToLower() } |
        Where-Object { $_.Length -gt 3 -and $_ -notin $stop } | Select-Object -Unique)
}

function Get-SharedTopicWords {
    param([string]$A, [string]$B)
    $wa = Get-TopicWords $A
    $wb = Get-TopicWords $B
    return @($wa | Where-Object { $wb -contains $_ })
}

function Read-Baseline {
    if (Test-Path $baselinePath) {
        try { return Get-Content $baselinePath -Raw | ConvertFrom-Json } catch { return $null }
    }
    return $null
}

function Send-Valeska {
    param([hashtable]$Payload, [string]$Path = '/api/v1/memory/thoughts')
    try {
        $restArgs = @{
            Uri         = "$ValeskaUrl$Path"
            Method      = 'Post'
            TimeoutSec  = 20
            ContentType = 'application/json'
            Headers     = @{ 'X-Valeska-Agent-Id' = $AgentId }
            Body        = ($Payload | ConvertTo-Json -Depth 8)
        }
        Invoke-RestMethod @restArgs
    } catch {
        # Valeska being down must never stop someone working. The local baseline
        # is what the pre-commit hook actually reads.
        Write-Host "  (Valeska unreachable  -  lease not filed, local guard still active)" -ForegroundColor Yellow
        return $null
    }
}

switch ($Action) {

    'start' {
        # Duplicate check FIRST, before a baseline is written or a worktree is
        # created. Isolation is what makes two agents invisible to each other,
        # so the moment to look is before you take any.
        $others = Get-ActiveLeases -ExcludeAgent $AgentId
        $clash = @()
        if ($others.Count -gt 0) {
            Write-Host ''
            Write-Host "  $($others.Count) other agent lease(s) OPEN on ${repoSlug}:" -ForegroundColor Yellow
            foreach ($o in $others) {
                $shared = if ($Task) { Get-SharedTopicWords $Task $o.summary } else { @() }
                $mark = if ($shared.Count -ge 2) { '  <-- OVERLAPS YOUR TASK'; } else { '' }
                Write-Host ("      {0}  {1}{2}" -f (Format-LeaseTime $o.created_at), $o.summary, $mark) `
                    -ForegroundColor $(if ($shared.Count -ge 2) { 'Red' } else { 'Gray' })
                if ($shared.Count -ge 2) { $clash += [pscustomobject]@{ lease = $o; shared = $shared } }
            }
            Write-Host '  Read these before you start. Duplication is judged on topic, not paths.' -ForegroundColor Yellow
        }

        if ($clash.Count -gt 0 -and -not $Force) {
            Write-Host ''
            Write-Host '  STOPPED - another agent is already on this topic.' -ForegroundColor Red
            foreach ($c in $clash) {
                Write-Host ("      {0}: shared terms [{1}]" -f $c.lease.agent_id, ($c.shared -join ', ')) -ForegroundColor Red
            }
            Write-Host ''
            Write-Host '  Do one of these instead of restarting their work:' -ForegroundColor Cyan
            Write-Host '    1. Read their artifact and extend it; say so in your task sentence.'
            Write-Host '    2. Ask Dave which of you owns it.'
            Write-Host '    3. Narrow your task so it genuinely does not overlap, and reword -Task.'
            Write-Host ''
            Write-Host '  If you are certain it is separate work, re-run with -Force.' -ForegroundColor Cyan
            Write-Host '  The override is recorded in your lease. See E:\REPOS\MULTI-AGENT-RULES.md section 3.'
            Write-Host ''
            exit 2
        }
        if ($clash.Count -gt 0 -and $Force) {
            Write-Host ''
            Write-Host '  -Force: proceeding despite an overlapping lease. Recorded.' -ForegroundColor Yellow
        }

        $dirty = Get-DirtyPaths
        $head = (git rev-parse HEAD 2>$null)
        $branch = (git rev-parse --abbrev-ref HEAD 2>$null)
        $now = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

        $baseline = [ordered]@{
            agent_id       = $AgentId
            task           = $Task
            branch         = $branch
            base_commit    = $head
            started_at     = $now
            # Everything already modified. The hook treats these as somebody
            # else's until claimed.
            dirty_at_start = @($dirty)
            claimed        = @()
            exclusive_paths = @($Paths -split ',' | Where-Object { $_ } | ForEach-Object { $_.Trim() })
        }
        # WriteAllText, not Set-Content -Encoding utf8: on Windows PowerShell 5.1
        # that writes a byte-order mark, which the hook's JSON reader used to
        # choke on - disabling the guard without a word.
        [System.IO.File]::WriteAllText($baselinePath, ($baseline | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))

        Write-Host ''
        Write-Host "Session started: $AgentId on $branch" -ForegroundColor Cyan
        Write-Host "  base commit : $head"
        if ($Task) { Write-Host "  task        : $Task" }
        if ($dirty.Count -gt 0) {
            Write-Host ''
            Write-Host "  $($dirty.Count) file(s) were ALREADY modified before you started:" -ForegroundColor Yellow
            $dirty | Select-Object -First 12 | ForEach-Object { Write-Host "      $_" }
            if ($dirty.Count -gt 12) { Write-Host "      ... and $($dirty.Count - 12) more" }
            Write-Host ''
            Write-Host '  Someone else is probably mid-task. You will be blocked from' -ForegroundColor Yellow
            Write-Host '  committing these unless you claim them.' -ForegroundColor Yellow
        } else {
            Write-Host '  working tree clean  -  nothing to collide with' -ForegroundColor Green
        }

        # Isolation decision. Dirty tree means somebody is mid-task here.
        $worktreePath = $null
        $wantWorktree = $Worktree -or ($dirty.Count -gt 0 -and -not $NoWorktree)
        if ($wantWorktree) {
            $slugSource = if ($Task) { $Task } else { 'work' }
            $slug = ($AgentId + '-' + ($slugSource -replace '[^a-zA-Z0-9]+', '-')).ToLower().Trim('-')
            if ($slug.Length -gt 48) { $slug = $slug.Substring(0, 48).Trim('-') }
            Write-Host ''
            Write-Host '  Creating an isolated worktree so you cannot collide with them...' -ForegroundColor Cyan
            $worktreePath = New-AgentWorktree -RepoRoot $repoRoot -Slug $slug
            if ($worktreePath) {
                $baseline.worktree_path = $worktreePath
                # WriteAllText, not Set-Content -Encoding utf8: on Windows PowerShell 5.1
        # that writes a byte-order mark, which the hook's JSON reader used to
        # choke on - disabling the guard without a word.
        [System.IO.File]::WriteAllText($baselinePath, ($baseline | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))
                Write-Host ''
                Write-Host '  WORK HERE INSTEAD:' -ForegroundColor Green
                Write-Host "      $worktreePath" -ForegroundColor Green
                Write-Host '  Own branch, own files. Merge back when you are done; git handles'
                Write-Host '  the overlap as a merge rather than one of you overwriting the other.'
            }
        }

        $lease = [ordered]@{
            lease_version   = '1.0'
            repo_slug       = $repoSlug
            repo_path_stable = ($repoRoot -replace '\\', '/')
            agent_id        = $AgentId
            task_summary    = $Task
            branch          = $branch
            worktree_path   = ($repoRoot -replace '\\', '/')
            status          = 'active'
            exclusive_paths = @($baseline.exclusive_paths)
            base_commit     = $head
            started_at      = $now
            dirty_at_start  = @($dirty)
            # Empty unless this session started on top of somebody else's open
            # lease. Kept so the override is auditable rather than invisible.
            forced_over     = @($clash | ForEach-Object { "$($_.lease.agent_id): $($_.lease.summary)" })
        }
        $null = Send-Valeska -Payload @{
            content          = "REPO LEASE (active): $AgentId on $repoSlug/$branch  -  $Task`n`n" + ($lease | ConvertTo-Json -Depth 6)
            thought_type     = 'memory'
            project          = $repoSlug
            agent_id         = $AgentId
            tags             = @('repo-lease', "repo:$repoSlug", "agent:$AgentId", 'active')
            visibility_scope = 'shared_team'
        }
        $factOk = Write-LeaseFact -Value @{
            status      = 'active'
            task        = $Task
            agent_id    = $AgentId
            branch      = $branch
            worktree    = $(if ($worktreePath) { $worktreePath -replace '\\', '/' } else { $repoRoot -replace '\\', '/' })
            base_commit = $head
            started_at  = $now
            exclusive_paths = @($baseline.exclusive_paths)
            forced_over = @($baseline.forced_over)
        }
        Write-Host ''
        if ($factOk) {
            Write-Host "  lease filed to Valeska  ->  $leaseKey" -ForegroundColor Green
        } else {
            Write-Host '  lease thought filed; keyed fact FAILED - other agents may not' -ForegroundColor Yellow
            Write-Host '  see this lease. Say so in your closeout.' -ForegroundColor Yellow
        }
        Write-Host ''
    }

    'claim' {
        $baseline = Read-Baseline
        if (-not $baseline) { Write-Host 'No session. Run: agent-session.ps1 start' -ForegroundColor Red; exit 1 }
        if (-not $Paths) { Write-Host 'Nothing to claim. Pass -Paths.' -ForegroundColor Red; exit 1 }

        $new = @($Paths -split ',' | Where-Object { $_ } | ForEach-Object { $_.Trim() -replace '\\', '/' })
        $baseline.claimed = @($baseline.claimed) + $new | Select-Object -Unique
        # WriteAllText, not Set-Content -Encoding utf8: on Windows PowerShell 5.1
        # that writes a byte-order mark, which the hook's JSON reader used to
        # choke on - disabling the guard without a word.
        [System.IO.File]::WriteAllText($baselinePath, ($baseline | ConvertTo-Json -Depth 6), (New-Object System.Text.UTF8Encoding($false)))

        Write-Host "Claimed $($new.Count) path(s); they will no longer block a commit:" -ForegroundColor Green
        $new | ForEach-Object { Write-Host "    $_" }
    }

    'status' {
        $baseline = Read-Baseline
        Write-Host ''
        if ($baseline) {
            Write-Host "Session: $($baseline.agent_id) since $($baseline.started_at)" -ForegroundColor Cyan
            Write-Host "  branch      : $($baseline.branch)"
            Write-Host "  base commit : $($baseline.base_commit)"
            $guarded = @($baseline.dirty_at_start | Where-Object { $_ -notin @($baseline.claimed) })
            Write-Host "  guarded     : $($guarded.Count) file(s) treated as someone else's"
            $guarded | Select-Object -First 10 | ForEach-Object { Write-Host "      $_" }
            Write-Host "  claimed     : $(@($baseline.claimed).Count)"
        } else {
            Write-Host 'No session recorded. The pre-commit guard is INACTIVE.' -ForegroundColor Yellow
            Write-Host 'Start one with: scripts\agent-session.ps1 start -Task "..."'
        }

        Write-Host ''
        Write-Host 'Other active leases in Valeska:' -ForegroundColor Cyan
        try {
            $body = @{ query = "REPO LEASE active $repoSlug"; project = $repoSlug; limit = 8 } | ConvertTo-Json
            $searchArgs = @{
                Uri         = "$ValeskaUrl/api/v1/memory/thoughts/search"
                Method      = 'Post'
                ContentType = 'application/json'
                TimeoutSec  = 20
                Headers     = @{ 'X-Valeska-Agent-Id' = $AgentId }
                Body        = $body
            }
            $res = Invoke-RestMethod @searchArgs
            $leases = @($res.thoughts | Where-Object { $_.tags -contains 'repo-lease' -and $_.tags -contains 'active' })
            if ($leases.Count -eq 0) { Write-Host '  none' }
            else {
                foreach ($l in $leases) {
                    $firstLine = ($l.content -split [Environment]::NewLine)[0]
                    if (-not $firstLine) { $firstLine = ($l.content -split "`n")[0] }
                    Write-Host ("  {0}  {1}  {2}" -f (Format-LeaseTime $l.created_at), $l.agent_id, $firstLine)
                }
            }
        } catch {
            Write-Host '  (Valeska unreachable)' -ForegroundColor Yellow
        }
        Write-Host ''
    }

    'scan' {
        # Read-only. Answers "is anyone else in here?" without touching a thing.
        #
        # The reason this exists: `git worktree list` marks a worktree prunable
        # whenever its folder is unreachable from THIS machine. Worktrees created
        # on santos live on santos's local disk while their bookkeeping syncs
        # through the shared .git, so stable sees live work as stale. On
        # 2026-08-09 that nearly cost ~2,000 lines of uncommitted work.
        Write-Host ''
        Write-Host "Worktrees registered for ${repoSlug}:" -ForegroundColor Cyan

        $wt = @(); $cur = $null
        foreach ($line in (git worktree list --porcelain 2>$null)) {
            if ($line -like 'worktree *') {
                if ($cur) { $wt += $cur }
                $cur = [ordered]@{ path = $line.Substring(9); branch = ''; prunable = $false }
            } elseif ($line -like 'branch *' -and $cur) {
                $cur.branch = $line.Substring(7) -replace '^refs/heads/', ''
            } elseif ($line -like 'prunable*' -and $cur) { $cur.prunable = $true }
        }
        if ($cur) { $wt += $cur }

        $unreachable = @()
        foreach ($w in $wt) {
            $winPath = $w.path -replace '/', '\'
            $here = Test-Path $winPath
            $label = if ($here) { 'reachable  ' } else { 'NOT ON THIS MACHINE' }
            $colour = if ($here) { 'Green' } else { 'Yellow' }
            Write-Host ("  [{0}] {1}" -f $label, $w.path) -ForegroundColor $colour
            if ($w.branch) { Write-Host ("                  branch: {0}" -f $w.branch) }
            if (-not $here) { $unreachable += $w }
        }

        if ($unreachable.Count -gt 0) {
            Write-Host ''
            Write-Host '  DO NOT PRUNE. Unreachable does not mean abandoned.' -ForegroundColor Red
            Write-Host '  Those paths are very likely alive on another machine. To check:' -ForegroundColor Yellow
            foreach ($w in $unreachable) {
                if ($w.path -match '^/home/([^/]+)/') {
                    Write-Host ("      ssh dave@santos `"ls -la {0}`"" -f $w.path)
                } else {
                    Write-Host ("      inspect: {0}" -f $w.path)
                }
            }
            Write-Host '  Then search Valeska for a lease from the owning agent.'
            Write-Host '  Only Dave may authorise `git worktree prune` or `remove`.' -ForegroundColor Yellow
        }

        Write-Host ''
        Write-Host 'Other agents with open leases:' -ForegroundColor Cyan
        $others = Get-ActiveLeases -ExcludeAgent $AgentId
        if ($others.Count -eq 0) { Write-Host '  none' }
        else { foreach ($o in $others) { Write-Host ("  {0}  {1}  {2}" -f (Format-LeaseTime $o.created_at), $o.agent_id, $o.summary) } }
        Write-Host ''
    }

    'end' {
        $baseline = Read-Baseline
        if ($baseline) {
            $null = Send-Valeska -Payload @{
                content          = "REPO LEASE (closed): $($baseline.agent_id) on $repoSlug/$($baseline.branch)  -  $($baseline.task)"
                thought_type     = 'memory'
                project          = $repoSlug
                agent_id         = $AgentId
                tags             = @('repo-lease', "repo:$repoSlug", "agent:$AgentId", 'closed')
                visibility_scope = 'shared_team'
            }
            # Close the keyed fact too, or the next agent to run `start` is
            # stopped by a lease nobody is holding any more.
            $null = Write-LeaseFact -Value @{
                status = 'closed'; task = "$($baseline.task)"; agent_id = $AgentId
                branch = "$($baseline.branch)"; closed_at = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
            }
            Remove-Item $baselinePath -Force -ErrorAction SilentlyContinue
            Write-Host 'Session closed. Guard removed; nothing was committed or cleaned.' -ForegroundColor Green
            Write-Host ''
            Write-Host 'Leave a receipt: commit, branch, stash, patch, or a note naming the' -ForegroundColor Cyan
            Write-Host 'exact paths left dirty. Never report done on unexplained dirty state.' -ForegroundColor Cyan
        } else {
            Write-Host 'No session to close.' -ForegroundColor Yellow
        }
    }
}
