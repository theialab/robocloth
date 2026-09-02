"""Stub evaluator behind tests/bonn_driver/stub_eval.sh.

Writes $OUTPUT_ROOT/eval_results_bonn/<mat>_<tag>.json through the real
collect_eval_results.write_result (so the provenance schema is the production
one) with the paper value of the cell plus an optional offset, and records the
same configuration identifiers evaluate_cell in run_table_bonn.sh records
(experiment=stage2_bonn / stage2_bonn_pbr, dataset_folder=$DATA_ROOT,
overrides, and the Bonn extras: component PSNRs and the 27/106 split counts)
— the driver demands the identifiers through `--reuse-check ... --expect`.

Env knobs (all optional):
  STUB_PSNR_OFFSETS        JSON {"mat/model": dB} added to the paper value
  STUB_OVERRIDES           space-separated overrides to record (default: none)
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
sys.path.insert(0, os.path.join(SCRIPTS, "comparisons"))
from collect_eval_results import write_result  # noqa: E402
from collect_eval_results_bonn import PAPER, bonn_val_psnr  # noqa: E402

mat, ckpt, tag = sys.argv[1:4]
cell = f"{mat}/{tag}"
data_root = os.environ["DATA_ROOT"]              # exported by the driver, like OUTPUT_ROOT
experiment = "stage2_bonn_pbr" if tag == "PBR" else "stage2_bonn"
N_POLY, N_GRAY = 27, 106                          # the held-out split of every table material


def cells(var):
    return os.environ.get(var, "").split()


if cell in cells("STUB_FAIL_CELLS"):
    print(f"[stub_eval] simulated evaluation failure for {cell}", file=sys.stderr)
    sys.exit(17)
if cell in cells("STUB_NO_JSON_CELLS"):
    print(f"[stub_eval] {cell}: exiting 0 without writing a result")
    sys.exit(0)

offsets = json.loads(os.environ.get("STUB_PSNR_OFFSETS", "{}"))
target = PAPER[mat][tag] + float(offsets.get(cell, 0.0))
poly = target - 2.0                               # any pair whose weighted mean is the target
gray = (target * (N_POLY + N_GRAY) - N_POLY * poly) / N_GRAY
psnr = bonn_val_psnr(poly, gray, N_POLY, N_GRAY)
recorded_ckpt = ckpt
if cell in cells("STUB_WRONG_CKPT_CELLS"):
    recorded_ckpt = os.path.abspath(__file__)          # exists, but is not the checkpoint
if cell in cells("STUB_WRONG_CONFIG_CELLS"):
    experiment = "stage2_debug"                        # not the config the driver asked for

results_dir = os.path.join(os.environ["OUTPUT_ROOT"], "eval_results_bonn")
os.makedirs(results_dir, exist_ok=True)
write_result(os.path.join(results_dir, f"{mat}_{tag}.json"), material=mat, model=tag,
             ckpt=recorded_ckpt, val_psnr=psnr, val_loss=None,
             experiment=experiment, dataset_folder=data_root, exp_name=f"stub_{mat}_{tag}",
             overrides=cells("STUB_OVERRIDES"), val_poly_psnr=poly, val_gray_psnr=gray,
             val_all_psnr=gray, n_val_poly=N_POLY, n_val_gray=N_GRAY, n_val_views=N_POLY + N_GRAY)
print(f"[stub_eval] {cell}: val psnr = {psnr:.2f} dB")
