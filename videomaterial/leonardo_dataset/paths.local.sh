#!/usr/bin/env bash
# dec-010: this worktree (branch videomaterial/b3-joint-sequences, the B3 dataset render) gets
# its own remote code dir on Leonardo, so it cannot collide with the B1/B2 dataset code
# (robocloth-synthetic-sequences), the benchmark tree (robocloth-leonardo-render) or any other
# agent's experiment. Sourced at the end of paths.sh; scripts/sync_code.sh reads it too.
export VM_REPO_ROOT="$WORK/zli00003/VideoMaterial/code/robocloth-b3-joint"
