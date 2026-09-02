"""Stub evaluator behind tests/repro_drivers/stub_eval.sh.

Writes $OUTPUT_ROOT/<results_subdir>/<mat>_<tag>.json through the real
collect_eval_results.write_result (so the provenance schema is the production
one) with the paper value of the cell plus an optional offset, and records the
same configuration identifiers eval_stage2.sh / eval_stage2_ubo.sh record
(experiment, dataset_folder [+ btf_path], overrides=[]) — the drivers demand
them through `--reuse-check ... --expect`.

Env knobs (all optional):
  STUB_TABLE               ours|ubo -> paper table + results subdir (default: ours)
  STUB_PSNR_OFFSETS        JSON {"mat/model": dB} added to the paper value
  STUB_FAIL_CELLS          space-separated "mat/model" cells that exit 17 without writing
  STUB_NO_JSON_CELLS       cells that exit 0 without writing any result
  STUB_WRONG_CKPT_CELLS    cells whose result records a different checkpoint file
  STUB_WRONG_CONFIG_CELLS  cells whose result records another experiment config
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, "scripts"))
sys.path.insert(0, SCRIPTS)
from collect_eval_results import write_result  # noqa: E402

mat, ckpt, tag = sys.argv[1:4]
cell = f"{mat}/{tag}"
data_root = os.environ["DATA_ROOT"]              # exported by the drivers, like OUTPUT_ROOT

if os.environ.get("STUB_TABLE", "ours") == "ubo":
    sys.path.insert(0, os.path.join(SCRIPTS, "comparisons"))
    from collect_eval_results_ubo import PAPER  # noqa: E402
    subdir = "eval_results_ubo"
    experiment = "eval_ubo_pbr" if tag == "PBR" else "eval_ubo"
    dataset = dict(dataset_folder=data_root, btf_path=os.path.join(data_root, f"{mat}_W400xH400_L151xV151.btf"))
else:
    from collect_eval_results import PAPER  # noqa: E402
    subdir = "eval_results"
    experiment = "eval_stage2_pbr" if tag == "PBR" else "eval_stage2"
    dataset = dict(dataset_folder=os.path.join(data_root, mat))


def cells(var):
    return os.environ.get(var, "").split()


if cell in cells("STUB_FAIL_CELLS"):
    print(f"[stub_eval] simulated evaluation failure for {cell}", file=sys.stderr)
    sys.exit(17)
if cell in cells("STUB_NO_JSON_CELLS"):
    print(f"[stub_eval] {cell}: exiting 0 without writing a result")
    sys.exit(0)

offsets = json.loads(os.environ.get("STUB_PSNR_OFFSETS", "{}"))
psnr = PAPER[mat][tag] + float(offsets.get(cell, 0.0))
recorded_ckpt = ckpt
if cell in cells("STUB_WRONG_CKPT_CELLS"):
    recorded_ckpt = os.path.abspath(__file__)          # exists, but is not the checkpoint
if cell in cells("STUB_WRONG_CONFIG_CELLS"):
    experiment = "stage2_debug"                        # not the config the driver asked for

results_dir = os.path.join(os.environ["OUTPUT_ROOT"], subdir)
os.makedirs(results_dir, exist_ok=True)
write_result(os.path.join(results_dir, f"{mat}_{tag}.json"), material=mat, model=tag,
             ckpt=recorded_ckpt, val_psnr=psnr, val_loss=0.01,
             experiment=experiment, exp_name=f"stub_{mat}_{tag}", overrides=[], **dataset)
print(f"[stub_eval] {cell}: val/psnr = {psnr:.2f} dB")
