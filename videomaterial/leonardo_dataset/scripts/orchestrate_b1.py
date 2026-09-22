#!/usr/bin/env python3
"""Drive the exp-015 B1 render end to end from g00s: submit, poll, resubmit once, rsync, verify.

Started through ``orchestrate_b1.sh`` under nohup/setsid so it outlives the agent session.
Everything it knows is in ``<state-dir>/status.json``; when it stops it leaves ``DONE`` or
``FAILED`` there, which is also the signal the quota monitor watches for.

It never deletes or modifies anything: it writes only under the state dir and rsyncs (without
--delete) into the local B1 tree.
"""
from __future__ import annotations

import argparse, json, os, shlex, subprocess, sys, time
from pathlib import Path

LOGIN = "zli00003@login.leonardo.cineca.it"
DATA = "zli00003@data.leonardo.cineca.it"
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=30"]
ENV = {**os.environ, "SSH_AUTH_SOCK": os.environ.get("SSH_AUTH_SOCK", "/tmp/vm-agent.sock")}
ACTIVE = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "COMPLETING"}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def ssh(cmd, timeout=300, check=True):
    p = subprocess.run(["ssh", *SSH_OPTS, LOGIN, cmd], capture_output=True, text=True,
                       timeout=timeout, env=ENV)
    if check and p.returncode != 0:
        raise RuntimeError(f"ssh failed ({p.returncode}): {cmd}\n{p.stderr.strip()}")
    return p.stdout.strip()


class Status:
    def __init__(self, path: Path, **init):
        self.path = path
        self.d = {"updated_utc": None, "state": "starting", "job_ids": [], "resubmitted": False,
                  "tasks": {}, "sequences_complete": 0, "message": "", **init}
        self.write()

    def update(self, **kw):
        self.d.update(kw)
        self.write()

    def write(self):
        self.d["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=2) + "\n")
        tmp.replace(self.path)


def remote_paths(repo_root):
    """Resolve the Leonardo-side paths once, by sourcing paths.sh remotely."""
    script = (f"source {repo_root}/videomaterial/leonardo_dataset/paths.sh && "
              'echo "$VM_B1_ROOT"; echo "$VM_RUN_ROOT"; echo "$VM_MANIFEST"')
    out = ssh("bash -lc " + shlex.quote(script))
    b1, run, man = out.splitlines()[:3]
    return b1, run, man


def submit(repo_root, run_root, array):
    script = (f"mkdir -p {run_root}/logs && cd {repo_root} && sbatch --parsable "
              f"--array={array} "
              f"--output={run_root}/logs/render-%A_%a.out "
              f"--error={run_root}/logs/render-%A_%a.err "
              "videomaterial/leonardo_dataset/jobs/render_b1_array.slurm")
    jid = ssh("bash -lc " + shlex.quote(script)).splitlines()[-1].strip()
    if not jid.split("_")[0].isdigit():
        raise RuntimeError(f"unexpected sbatch output: {jid!r}")
    return jid


def task_states(job_id):
    out = ssh(f"sacct -j {job_id} -X -n -P -o JobID,State,Elapsed", check=False)
    st = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 2 or "_" not in parts[0]:
            continue
        tid = parts[0].split("_", 1)[1]
        if tid.startswith("["):          # a pending array range, e.g. 12345_[4-15]
            body = tid.strip("[]").split("%")[0]
            for chunk in body.split(","):
                if "-" in chunk:
                    a, b = chunk.split("-")
                    for i in range(int(a), int(b) + 1):
                        st[str(i)] = parts[1].split()[0]
                elif chunk.isdigit():
                    st[chunk] = parts[1].split()[0]
        else:
            st[tid] = parts[1].split()[0]
    return st


def complete_dirs(b1_root):
    out = ssh(f"find {b1_root}/train {b1_root}/test -maxdepth 2 -name RENDER_COMPLETE -printf '%h\\n' 2>/dev/null || true",
              timeout=600, check=False)
    return {line.strip().replace(b1_root.rstrip("/") + "/", "") for line in out.splitlines() if line.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--local-root", type=Path, required=True)
    ap.add_argument("--repo-root", default="/leonardo_work/IscrB_OVER/zli00003/VideoMaterial/code/robocloth-synthetic-sequences")
    ap.add_argument("--nshards", type=int, default=16)
    ap.add_argument("--array", default="0-15")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--no-submit", action="store_true", help="attach to the job ids already in status.json")
    args = ap.parse_args()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads(args.manifest.read_text())
    seqs = man["sequences"]
    status = Status(args.state_dir / "status.json", sequences_total=len(seqs), nshards=args.nshards,
                    manifest=str(args.manifest), local_root=str(args.local_root))
    for flag in ("DONE", "FAILED"):
        (args.state_dir / flag).unlink(missing_ok=True)

    b1_root, run_root, remote_manifest = remote_paths(args.repo_root)
    log(f"remote B1 root {b1_root}; run root {run_root}")
    status.update(remote_b1_root=b1_root, remote_run_root=run_root, state="submitting")

    baseline = ssh("saldo -b", check=False)
    (args.state_dir / "saldo_baseline.txt").write_text(baseline + "\n")

    if args.no_submit:
        job_ids = status.d["job_ids"]
    else:
        jid = submit(args.repo_root, run_root, args.array)
        log(f"submitted array job {jid} ({args.array})")
        job_ids = [jid]
    status.update(job_ids=job_ids, state="running")

    fails = 0
    while True:
        try:
            st = {}
            for j in job_ids:
                st.update({f"{j}_{k}": v for k, v in task_states(j).items()})
            done_dirs = complete_dirs(b1_root)
            fails = 0
        except Exception as e:                       # transient ssh trouble must not end the run
            fails += 1
            log(f"poll failed ({fails}): {e}")
            status.update(message=f"poll failure {fails}: {e}")
            if fails >= 12:
                (args.state_dir / "FAILED").write_text(f"12 consecutive poll failures: {e}\n")
                status.update(state="failed", message="12 consecutive poll failures")
                return 1
            time.sleep(60)
            continue

        active = sum(1 for v in st.values() if v in ACTIVE)
        status.update(state="running", tasks=st, sequences_complete=len(done_dirs),
                      task_counts={s: sum(1 for v in st.values() if v == s) for s in sorted(set(st.values()))},
                      message=f"{len(done_dirs)}/{len(seqs)} sequences complete, {active} tasks active")
        log(f"{len(done_dirs)}/{len(seqs)} sequences complete; tasks: " +
            ", ".join(f"{s}={n}" for s, n in status.d['task_counts'].items()))

        if active == 0 and st:
            missing = [s for s in seqs if s["dir"] not in done_dirs]
            if not missing:
                break
            shards = sorted({s["index"] % args.nshards for s in missing})
            if status.d["resubmitted"]:
                log(f"{len(missing)} sequences still missing after the resubmit; stopping")
                break
            log(f"{len(missing)} sequences missing in shards {shards}; resubmitting once")
            jid = submit(args.repo_root, run_root, ",".join(map(str, shards)))
            job_ids.append(jid)
            status.update(job_ids=job_ids, resubmitted=True,
                          message=f"resubmitted shards {shards} as {jid}")
        time.sleep(args.poll)

    # ---------------------------------------------------------------- transfer
    status.update(state="rsync", message="copying from Leonardo")
    args.local_root.mkdir(parents=True, exist_ok=True)
    rsync_log = (args.state_dir / "rsync.log").open("a", buffering=1)
    for sub in ("manifest.json", "train", "test", "logs"):
        src = f"{DATA}:{b1_root}/{sub}"
        dst = str(args.local_root) + "/"
        cmd = ["rsync", "-a", "--partial", "--info=stats2", "-e", "ssh " + " ".join(SSH_OPTS), src, dst]
        log("rsync " + sub)
        p = subprocess.run(cmd, env=ENV, stdout=rsync_log, stderr=subprocess.STDOUT)
        if p.returncode != 0 and sub != "logs":
            (args.state_dir / "FAILED").write_text(f"rsync of {sub} failed with {p.returncode}\n")
            status.update(state="failed", message=f"rsync {sub} rc={p.returncode}")
            return 1

    # ---------------------------------------------------------------- verify
    status.update(state="verify", message="checking MANIFEST.sha256 locally")
    ok = bad = absent = 0
    bad_dirs = []
    for s in seqs:
        d = args.local_root / s["dir"]
        if not (d / "MANIFEST.sha256").exists():
            absent += 1
            bad_dirs.append(f"{s['dir']}: no MANIFEST.sha256")
            continue
        p = subprocess.run(["sha256sum", "-c", "--quiet", "MANIFEST.sha256"], cwd=d,
                           capture_output=True, text=True)
        if p.returncode == 0:
            ok += 1
        else:
            bad += 1
            bad_dirs.append(f"{s['dir']}: {p.stdout.strip()[:200]}")
    (args.state_dir / "verify_failures.txt").write_text("\n".join(bad_dirs) + "\n")
    counts = {"sequences_total": len(seqs), "checksum_ok": ok, "checksum_bad": bad, "missing_locally": absent}
    status.update(counts=counts)
    log(f"verification: {counts}")

    if bad == 0 and absent == 0:
        (args.state_dir / "DONE").write_text(json.dumps(counts, indent=2) + "\n")
        status.update(state="done", message="all sequences transferred and verified")
        return 0
    (args.state_dir / "FAILED").write_text(json.dumps(counts, indent=2) + "\n")
    status.update(state="failed", message=f"{bad} checksum failures, {absent} missing locally")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
