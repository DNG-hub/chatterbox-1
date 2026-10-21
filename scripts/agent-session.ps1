# Shim: the real guard lives in E:\REPOS\agent-session.ps1 (one copy, no drift). Run from inside the repo.
$root = if ($env:REPOS_ROOT) { $env:REPOS_ROOT } else { 'E:\REPOS' }
& (Join-Path $root 'agent-session.ps1') @args; if ($?) { exit 0 } elseif ($LASTEXITCODE) { exit $LASTEXITCODE } else { exit 1 }
