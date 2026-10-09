#!/bin/bash
# usage: run_queue.sh GPU QUEUEFILE   -- worker: pops one job line at a time (flock), runs it
# through run_batch.sh; several workers (GPUs) share one queue and balance themselves.
D=$(dirname "$(readlink -f "$0")")
Q=$2
while true; do
  job=$(flock "$Q.lock" bash -c 'j=$(head -n 1 "$0"); [ -n "$j" ] && sed -i 1d "$0"; echo "$j"' "$Q")
  [ -z "$job" ] && break
  "$D/run_batch.sh" "$1" "$job" | grep -v '^DONE'
done
echo "WORKER $1 DONE $(date +%T)"
