#!/usr/bin/env bash
# exp-015 B1 quota guard, run on g00s alongside the orchestrator.
#
#   VM_EXPECTED_LOCAL_H=<measured local-h per sequence x 1000> \
#   nohup setsid bash videomaterial/leonardo_dataset/scripts/quota_monitor.sh \
#       > /media/raid/cloth/VideoMaterial/output/synthetic_experiments/exp15_b1_render/quota_monitor.log 2>&1 &
#
# Every hour it reads `saldo -b` over SSH and appends the consumed local-h for
# IscrB_OVER to quota.log. If the delta since the baseline recorded at start
# exceeds 1.5x the expected total, it scancels ONLY the job ids listed in
# status.json (nothing else, ever) and writes ALERT into quota.log and status.json.
# It also exits when the orchestrator writes DONE or FAILED.
#
# Stop it with:  pkill -f quota_monitor
set -uo pipefail

export SSH_AUTH_SOCK="${SSH_AUTH_SOCK:-/tmp/vm-agent.sock}"
LOGIN="zli00003@login.leonardo.cineca.it"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=30)
ACCOUNT="${VM_ACCOUNT:-IscrB_OVER}"

STATE_DIR="${VM_STATE_DIR:-/media/raid/cloth/VideoMaterial/output/synthetic_experiments/exp15_b1_render}"
QUOTA_LOG="$STATE_DIR/quota.log"
STATUS="$STATE_DIR/status.json"
EXPECTED="${VM_EXPECTED_LOCAL_H:?VM_EXPECTED_LOCAL_H (expected total local-h for the run) must be set}"
INTERVAL="${VM_QUOTA_INTERVAL:-3600}"
LIMIT=$(python3 -c "print(f'{1.5 * $EXPECTED:.1f}')")

mkdir -p "$STATE_DIR"
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$QUOTA_LOG"; }

# `saldo -b` row for the account -> localClusterConsumed (local h), column 5: what this
# cluster has burned. Columns: account start end total localClusterConsumed totConsumed ...
consumed() {
    ssh "${SSH_OPTS[@]}" "$LOGIN" "saldo -b" 2>/dev/null \
        | awk -v a="$ACCOUNT" '$1 == a {print $5}' | head -1
}

BASELINE="$(consumed)"
if [ -z "$BASELINE" ]; then
    say "ALERT could not read the saldo baseline; monitor refuses to start blind"
    exit 1
fi
say "baseline totConsumed=$BASELINE local-h for $ACCOUNT; expected for this run=$EXPECTED; cancel above $LIMIT"

while true; do
    if [ -f "$STATE_DIR/DONE" ] || [ -f "$STATE_DIR/FAILED" ]; then
        NOW="$(consumed)"
        say "orchestrator finished ($( [ -f "$STATE_DIR/DONE" ] && echo DONE || echo FAILED )); final totConsumed=${NOW:-unknown} delta=$(python3 -c "print(f'{${NOW:-$BASELINE} - $BASELINE:.1f}')"); monitor exits"
        exit 0
    fi
    NOW="$(consumed)"
    if [ -z "$NOW" ]; then
        say "WARN saldo unreadable this cycle; will retry"
    else
        DELTA=$(python3 -c "print(f'{$NOW - $BASELINE:.1f}')")
        PCT=$(python3 -c "print(f'{100.0 * ($NOW - $BASELINE) / $EXPECTED:.1f}')")
        say "totConsumed=$NOW delta=$DELTA local-h ($PCT% of the expected $EXPECTED)"
        OVER=$(python3 -c "print(1 if ($NOW - $BASELINE) > $LIMIT else 0)")
        if [ "$OVER" = "1" ]; then
            JOBS=$(python3 -c "import json;print(' '.join(json.load(open('$STATUS'))['job_ids']))" 2>/dev/null || echo "")
            say "ALERT delta $DELTA local-h exceeds 1.5x expected ($LIMIT). Cancelling job ids: ${JOBS:-none}"
            for j in $JOBS; do
                ssh "${SSH_OPTS[@]}" "$LOGIN" "scancel $j" && say "ALERT scancel $j sent" || say "ALERT scancel $j FAILED"
            done
            python3 - <<PY || true
import json
p = "$STATUS"
try:
    d = json.load(open(p))
except Exception:
    d = {}
d["quota_alert"] = {"delta_local_h": $DELTA, "expected_local_h": $EXPECTED, "limit_local_h": $LIMIT,
                    "cancelled": "${JOBS:-}".split(), "utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
d["state"] = "quota_alert"
json.dump(d, open(p, "w"), indent=2)
PY
            say "ALERT written to status.json; monitor exits"
            exit 2
        fi
    fi
    sleep "$INTERVAL"
done
