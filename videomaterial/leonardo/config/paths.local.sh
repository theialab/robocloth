#!/usr/bin/env bash
# Machine-local override for this worktree (dec-010): the exp-015 benchmark gets
# its own remote code directory so it cannot race with any other agent's rsync.
export VM_REPO_ROOT="$WORK/zli00003/VideoMaterial/code/robocloth-leonardo-render"
