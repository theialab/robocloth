#!/usr/bin/env bash
# dec-010: this worktree gets its own remote code dir on Leonardo, so the B1 dataset
# render cannot collide with the benchmark tree (robocloth-leonardo-render) or with
# any other agent's experiment. Sourced at the end of paths.sh.
export VM_REPO_ROOT="$WORK/zli00003/VideoMaterial/code/robocloth-synthetic-sequences"
