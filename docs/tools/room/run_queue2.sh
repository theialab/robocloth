#!/bin/bash
# usage: run_queue2.sh GPU QUEUEFILE   -- worker like run_queue.sh, but checks every job's output:
# a job whose file is missing or not written during the job (OOM, crash) goes to
# QUEUEFILE.failed and is logged as FAILED (re-queue those by hand, or with --retry once).
D=$(dirname "$(readlink -f "$0")")
Q=$2
P3=/media/raid/cloth/output/teaser_video/room_demo/passes_r5
X=/media/raid/cloth/output/teaser_video/room_demo/variants/xtalk/renders
out_of() {
  IFS=: read -r st o m spp suf <<< "$1"
  case $st in
    neural) echo "$P3/neural_${o}_${m}${suf}.exr" ;;
    lo) [ "$o" = grey ] && echo "$P3/lo_grey.exr" || echo "$P3/lo_${o}_${m}.exr" ;;
    swatch) echo "$P3/swatch_${o}.exr" ;;
    xtalk) echo "$X/${o}.exr" ;;
    masks) echo "$P3/masks.npy" ;;
  esac
}
while true; do
  job=$(flock "$Q.lock" bash -c 'j=$(head -n 1 "$0"); [ -n "$j" ] && sed -i 1d "$0"; echo "$j"' "$Q")
  [ -z "$job" ] && break
  stamp=$(mktemp)
  "$D/run_batch.sh" "$1" "$job" | grep --line-buffered -v '^DONE'
  f=$(out_of "$job")
  if [ -n "$f" ] && [ "$f" -nt "$stamp" ]; then :; else
    echo "FAILED $job (no fresh $f) $(date +%T)"
    flock "$Q.lock" bash -c "echo '$job' >> '$Q.failed'"
  fi
  rm -f "$stamp"
done
echo "WORKER $1 DONE $(date +%T)"
