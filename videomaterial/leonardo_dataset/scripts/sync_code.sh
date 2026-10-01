#!/usr/bin/env bash
# Rsync this worktree to its own Leonardo code dir (dec-010; VM_REPO_ROOT from paths.local.sh)
# and stamp it: CODE_VERSION (branch / commit / dirty / synced_utc, read by seqlib.git_info, so
# every metadata.json records the commit) and SYNCED_COMMIT (the bare sha).
#
#   videomaterial/leonardo_dataset/scripts/sync_code.sh            # refuses a dirty worktree
#   ALLOW_DIRTY=1 videomaterial/leonardo_dataset/scripts/sync_code.sh
#
# Needs SSH_AUTH_SOCK with the Leonardo key (g00s: /tmp/vm-agent.sock). Stamps are staged in
# $TMPDIR (must be on the fileserver on g00s), never in the worktree.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_HOST="${VM_DATA_HOST:-zli00003@data.leonardo.cineca.it}"
REMOTE_WORK="${VM_REMOTE_WORK:-/leonardo_work/IscrB_OVER}"
REMOTE_DIR="$(WORK="$REMOTE_WORK" bash -c "source '$ROOT/videomaterial/leonardo_dataset/paths.local.sh'; echo \"\$VM_REPO_ROOT\"")"
case "$REMOTE_DIR" in */code/robocloth-*) ;; *) echo "refusing: unexpected remote code dir '$REMOTE_DIR'"; exit 1;; esac

commit="$(git -C "$ROOT" rev-parse HEAD)"
branch="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
dirty=false; [ -n "$(git -C "$ROOT" status --porcelain)" ] && dirty=true
if [ "$dirty" = true ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
    echo "worktree has uncommitted changes; commit first (or ALLOW_DIRTY=1)"; exit 1
fi

stage="$(mktemp -d "${TMPDIR:-/tmp}/vm_sync_XXXXXX")"
trap 'rm -rf "$stage"' EXIT
printf 'branch=%s\ncommit=%s\ndirty=%s\nsynced_utc=%s\n' "$branch" "$commit" "$dirty" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$stage/CODE_VERSION"
echo "$commit" > "$stage/SYNCED_COMMIT"

echo "sync $ROOT ($branch @ ${commit:0:7}, dirty=$dirty) -> $DATA_HOST:$REMOTE_DIR"
rsync -a --delete --exclude .git --exclude assets --exclude __pycache__ --exclude '*.pyc' \
    --exclude CODE_VERSION --exclude SYNCED_COMMIT "$ROOT/" "$DATA_HOST:$REMOTE_DIR/"
rsync -a "$stage/CODE_VERSION" "$stage/SYNCED_COMMIT" "$DATA_HOST:$REMOTE_DIR/"
echo "done: SYNCED_COMMIT=$commit"
