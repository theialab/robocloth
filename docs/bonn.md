# Bonn (UBOFAB19) comparison: data, training, evaluation

Everything needed to reproduce the Bonn block of Table 3 (bottom: cross-domain
on the held-out Bonn test materials 318, 377, 32, 226, 37) and to retrain the
Bonn decoder — dataset download, expected layout, metadata generation, the
stage-1 / stage-2 launchers, the table driver and the numbers it checks.

"Bonn" in the paper is the **UBOFAB19** SVBRDF fabric dataset of the
University of Bonn (Merzbach et al., 2019) — 311 training and 67 validation
fabrics measured with a camera dome (RGB "poly" images, panchromatic "pan"
images and linear-light-source "LLS" images per fabric). It is not the
UBO2014 BTF dataset used for Table 2 (that one is downloaded with
`scripts/comparisons/download_ubo2014.sh`, see
[reproduce_paper.md](reproduce_paper.md#comparison-datasets-merl-ubo2014-bonn)).
Neither dataset is part of our Hugging Face bundle; the Bonn data is
downloaded from the Bonn servers with the scripts under
`scripts/comparisons/bonn_data/` as described below.

Environment: the training environment of
[optimize_new_material.md](optimize_new_material.md) §1 (it includes `pyexr`
and the OpenEXR bindings that the loader and the metadata script use). Use
absolute paths everywhere — the launch scripts change working directories.

```bash
export CKPTS=/absolute/path/to/robocloth-checkpoints     # Hugging Face bundle
export BONN_VAL=/absolute/path/to/Bonn_val               # UBOFAB19 validation measurements
export BONN_TRAIN=/absolute/path/to/Bonn_train           # UBOFAB19 training measurements (stage 1 only)
export OUTPUT_ROOT=/absolute/path/to/robocloth-output
```

## 1. Download the data

Four list scripts under `scripts/comparisons/bonn_data/` mirror the four
download lists published at `https://cg.cs.uni-bonn.de/btf/UBOFAB19/`. Each
script fetches the list (`UBOFAB19_<split>_<kind>.txt`) and then every file
in it into the **current directory** (`wget -c`, resumable), so run each one
from inside its target folder:

| Script | Split / content | Files | Size | Needed for |
|---|---|---|---|---|
| `get_UBOFAB19_val_meas.sh` | validation **measurements**: 67 fabrics × 9 files | 603 | ≈ 57 GB | **evaluation** (the table driver) and stage-2 training on the test fabrics |
| `get_UBOFAB19_train_meas.sh` | training measurements: 311 fabrics × 9 files | 2,799 | ≈ 180 GB | stage-1 training of the Bonn decoder |
| `get_UBOFAB19_val_axf_svfresnel.sh` | validation AxF materials (`matXXXX_svfresnel.axf`) | 67 | small | optional (see below) |
| `get_UBOFAB19_train_axf_svfresnel.sh` | training AxF materials | 311 | small | optional |

```bash
mkdir -p "$BONN_VAL" && cd "$BONN_VAL" && bash /absolute/path/to/RoboCloth/scripts/comparisons/bonn_data/get_UBOFAB19_val_meas.sh
# stage-1 retraining only:
mkdir -p "$BONN_TRAIN" && cd "$BONN_TRAIN" && bash /absolute/path/to/RoboCloth/scripts/comparisons/bonn_data/get_UBOFAB19_train_meas.sh
```

**Evaluation needs only the validation measurements** — all five table
materials (`mat0032`, `mat0037`, `mat0226`, `mat0318`, `mat0377`) are in
`UBOFAB19_val_meas.txt`. The training measurements are needed only to
retrain the stage-1 Bonn decoder (the released `checkpoints/stage1/Bonn.ckpt`
is the result of that run). The AxF files are not read by any shipped
configuration: the loader can derive ground-truth normal maps from them
(`Bonn_svfresnel/matXXXX_svfresnel/matXXXX_svfresnel_Normal.exr`, decoded
with the AxF SDK), but every released model predicts its own frame
(`predict_frame: True`), so those normals are ignored even when present.

## 2. Expected directory structure

One flat folder per split — no per-material sub-folders. After the download
and the metadata step (§3) a validation folder looks like this:

```
/absolute/path/to/Bonn_val/
├── UBOFAB19_val_meas.txt            # the download list (left by the script)
├── bonn_point_metadata.json         # generated in §3; required by the models
├── mat0003.txt                      # name, fabric type, glossiness, Fresnel F0,
├── mat0003_calibration.mat          #   pixel and physical dimensions
├── mat0003_poly.exr                 # RGB images (100 per fabric), one 3-channel group per image
├── mat0003_pan.exr                  # panchromatic images (388; the 288 with LEDs il001-il024 are used)
├── mat0003_lls.exr                  # linear-light-source images (280)
├── mat0003_xyz_rot000.exr           # per-pixel 3-D surface points (defines H x W)
├── mat0003_xyz_others.exr           # \
├── mat0003_xyz_refined_rot000.exr   #  not read by this code base
├── mat0003_xyz_refined_others.exr   # /
├── mat0004.txt ...                  # 67 fabrics in Bonn_val, 311 in Bonn_train
```

Per fabric the loader (`training/datasets/bonn.py`) opens exactly
`matXXXX_calibration.mat`, `matXXXX_xyz_rot000.exr`, `matXXXX_poly.exr`,
`matXXXX_pan.exr` and `matXXXX_lls.exr`; the models read
`bonn_point_metadata.json`. The remaining files may stay in place. Material
ids are the numbers in the file names (`mat0318` = material 318); the image
resolution differs per fabric (318: 422 × 776 px, 377: 1035 × 1190 px, 32:
843 × 408, 226: 417 × 1663, 37: 375 × 1674). Every validation fabric
carries the same 668 images (100 poly + 288 pan + 280 LLS; checked over all
67 files of the released validation set).

Nothing else is needed — no `scan_log.json`, `rotated_camera.json` or
emitter calibration (those entries of `configs/data/bonn.yaml` are unused
placeholders inherited from the RoboCloth configs).

## 3. Generate the metadata

`bonn_point_metadata.json` lists `H`, `W` and `num_points = H·W` per material;
the models (`BonnLatentBRDF`, `BonnPBRLatentBRDF`) size the per-pixel latent
bank from it before a checkpoint is loaded, and the renderer's Bonn wrapper
reads the same file. Run once per downloaded folder (and again after adding
materials):

```bash
python scripts/comparisons/generate_bonn_metadata.py "$BONN_VAL"      # evaluation / stage 2
python scripts/comparisons/generate_bonn_metadata.py "$BONN_TRAIN"    # stage 1
```

It reads only the `matXXXX_xyz_rot000.exr` headers and writes, e.g.
`"318": {"H": 422, "W": 776, "num_points": 327472}`. The table driver refuses
to start if the file, one of its entries, or one of the five per-material
files above is missing (exit 2, every problem listed).

## 4. Released checkpoints

The bundle [koalapenguin/RoboCloth-assets](https://huggingface.co/datasets/koalapenguin/RoboCloth-assets)
contains exactly these Bonn checkpoints (layout
`checkpoints/stage2/Bonn/<mat>/<MODEL>_epoch<N>.ckpt`; the `Ours` file name
is the paper's **RoboCloth** column):

```bash
hf download koalapenguin/RoboCloth-assets --repo-type dataset \
    --include "checkpoints/stage2/Bonn/*" --include "checkpoints/stage1/*" --local-dir "$CKPTS"
```

| Material | Ours (RoboCloth) | Bonn | MERL | PBR | size per file |
|---|---|---|---|---|---|
| 318 | `Ours_epoch120.ckpt` | `Bonn_epoch120.ckpt` | `MERL_epoch120.ckpt` | `PBR_epoch120.ckpt` | 60 MB (PBR 36 MB) |
| 377 | `Ours_epoch100.ckpt` | `Bonn_epoch100.ckpt` | `MERL_epoch100.ckpt` | `PBR_epoch100.ckpt` | 224 MB (PBR 134 MB) |
| 32 | `Ours_epoch120.ckpt` | `Bonn_epoch120.ckpt` | `MERL_epoch120.ckpt` | `PBR_epoch120.ckpt` | 63 MB (PBR 37 MB) |
| 226 | `Ours_epoch120.ckpt` | `Bonn_epoch120.ckpt` | `MERL_epoch120.ckpt` | `PBR_epoch120.ckpt` | 126 MB (PBR 75 MB) |
| 37 | `Ours_epoch120.ckpt` | `Bonn_epoch120.ckpt` | `MERL_epoch120.ckpt` | `PBR_epoch120.ckpt` | 114 MB (PBR 68 MB) |

plus the stage-1 priors `checkpoints/stage1/{Ours,Bonn,MERL}.ckpt`
(`Bonn.ckpt` is 2.2 GB and is the decoder trained in §5.1).

**Naming.** `<MODEL>_epoch<N>.ckpt` is the Lightning checkpoint saved after
`N` completed epochs (Lightning's own `epoch` field inside is `N-1`; e.g.
`377/*_epoch100.ckpt` records `epoch=99, global_step=131800`, the
`*_epoch120.ckpt` files `epoch=119`). The driver requires exactly one
`<MODEL>_epoch*.ckpt` per material and model; a missing or duplicated file is
reported for every cell before anything runs.

**Contents.** A neural Bonn checkpoint holds 13 tensors: `pan_weights`
(RGB→gray projection), `material.factor` (the per-channel scale β),
`material.point_latent_bank.weight` (`H·W × 30`: 24 latent + 6 frame
dimensions) and the 10 decoder tensors (`material.decoder.mlp.*`). A PBR
checkpoint holds `pan_weights`, `material.factor`, an `H·W × 18` parameter
bank and the decoder's canonical frame buffers. Evaluation loads all of them
strictly (every tensor must exist with a matching shape).

## 5. Training

Hyperparameters live in `configs/experiment/stage1_bonn.yaml`,
`stage2_bonn.yaml` and `stage2_bonn_pbr.yaml`; the launchers take paths only.
Both stages dispatch to `Stage1Trainer_Bonn` (`training/trainers/__init__.py`)
— intentionally: the Bonn implementation uses one trainer for both phases
(stage 2 swaps in the single-material datasets and freezes the decoder).

### 5.1 Stage 1 — the Bonn decoder (the "Bonn" column)

```bash
DATA_ROOT="$BONN_TRAIN" OUTPUT_ROOT="$OUTPUT_ROOT" bash scripts/comparisons/train_stage1_bonn.sh
```

Trains the shared decoder plus per-pixel latents on all 311 training
fabrics (`stage1_bonn.yaml`: 100 epochs × 8,000 batches of 5×10⁵ rays,
8-bit Adam, log-relative loss, 10 % pixel subsampling per fabric,
grazing-angle regularisation). Validation uses material 1, which exists only
in `Bonn_train`. Output:
`$OUTPUT_ROOT/stage1_bonn/training/model_0.20_0.20/last.ckpt` — the
released `checkpoints/stage1/Bonn.ckpt`. `EXP_NAME` renames the run folder;
extra Hydra overrides can be appended.

### 5.2 Stage 2 — one held-out Bonn material

```bash
# decoder source: MODEL=Ours (RoboCloth prior) | Bonn | MERL, with the matching stage-1 checkpoint
DATA_ROOT="$BONN_VAL" OUTPUT_ROOT="$OUTPUT_ROOT" MODEL=Ours \
STAGE1_CKPT="$CKPTS/checkpoints/stage1/Ours.ckpt" \
    bash scripts/comparisons/train_stage2_bonn.sh 318

# Disney-PBR baseline (trained from scratch; no STAGE1_CKPT)
DATA_ROOT="$BONN_VAL" OUTPUT_ROOT="$OUTPUT_ROOT" MODEL=PBR \
    bash scripts/comparisons/train_stage2_bonn.sh 318
```

`stage2_bonn.yaml`: 120 epochs × 743 batches of 5×10⁵ rays, validation every
4 epochs, decoder frozen and loaded decoder-only from `STAGE1_CKPT` (10
tensors), learnable β, log-relative loss, cosine-weighted BRDF. The 668
images of the material are split 80/20 with a fixed seed
(`val_view_ratio: 0.2`, `val_seed: 42`): 535 training views, 133 held-out
views (27 poly + 106 pan/LLS — identical for every validation fabric, since
all of them carry the same 668 images).
Output folder `$OUTPUT_ROOT/stage2_bonn318_from_Ours/` (PBR:
`stage2_bonn318_from_PBR/`):

* `training/model_0.20_0.20/epoch=<k>.ckpt` every 4 epochs and `last.ckpt`;
* `images/gt_mat0318_view<i>_{poly,gray}.png` and
  `images/pred_mat0318_view<i>_{poly,gray}_psnr<PSNR>.png` for the first 20
  held-out views, `normal_mat0318.png`, `tangent_mat0318.png`;
* the CSV/W&B logs with `val/poly_psnr`, `val/gray_psnr`, `val/all_psnr`
  (W&B runs offline by default, `WANDB_MODE=offline`).

To evaluate your own run with the table driver, copy `last.ckpt` to
`<your CKPT_ROOT>/318/Ours_epoch120.ckpt` (any `<MODEL>_epoch<N>` name).
The 377 checkpoints in the bundle come from an earlier schedule (their stored
hyper-parameters say 100 epochs × 1,318 batches and `model.lls_spp: 4`); the
other four materials match `stage2_bonn.yaml` as shipped.

Hardware: a single large GPU (the runs behind the paper used 48–80 GB cards)
and a host with enough RAM for the material's pixel data (≈ 0.5–2 GB of EXR
per fabric, held in memory twice: training split and held-out views).

## 6. Evaluation

There is no separate `eval_*.sh` for Bonn; the evaluation of one
(material, model) cell is the `evaluate_cell` function of
`scripts/comparisons/run_table_bonn.sh`. It runs, from `training/`:

```bash
# +experiment=stage2_bonn_pbr for the PBR column
python train.py +experiment=stage2_bonn \
    dataset_folder="$BONN_VAL" data.overfit_mat_id=318 data.valid_num=20 \
    model.test=true model.trainer.enable_checkpointing=false \
    model.ckpt_path="$CKPTS/checkpoints/stage2/Bonn/318/Ours_epoch120.ckpt" \
    output_folder="$OUTPUT_ROOT" exp_output_root_path="$OUTPUT_ROOT/Eval_Bonn_318_Ours" \
    experiment_name=Eval_Bonn_318_Ours \
    'model.logger._target_=pytorch_lightning.loggers.CSVLogger' '~model.logger.project'
```

`model.test=true` on top of the stage-2 experiment is the same switch
`configs/experiment/eval_stage2.yaml` applies to `stage2.yaml` for RoboCloth
materials: `train.py` loads **all** weights of the checkpoint (only emitter
buffers are exempt; Bonn has none) and runs `Stage1Trainer_Bonn`'s
validation step once over the full held-out split — the 133 views above,
regardless of `data.valid_num`, which only limits how many GT/prediction
PNGs are written. Note that `train.py` also constructs the training split in
test mode (Bonn branch), so the material's EXR files are read twice; expect
a few minutes per cell dominated by loading, and roughly 7 GB (318) to
25 GB (377) of host RAM for the precomputed held-out views.

**The metric.** For every held-out view the trainer computes a PSNR with
fixed peak 1.0 over the valid pixels (`10·log10(1/MSE)`, no clamping):
3-channel for poly views, 1-channel for pan and LLS views after the
calibrated RGB→gray projection (LLS views are Monte-Carlo integrated over
the light source with `model.lls_spp` samples). It logs the per-view mean
per kind, `val/poly_psnr` and `val/gray_psnr`. The paper number is their
**image-count-weighted mean**, `(27·poly + 106·gray) / 133` — i.e. the plain
mean per-view PSNR over all held-out views (every view has the same pixel
count). The driver recomputes the two counts from the material's EXR
headers with the loader's own channel parsers and split recipe
(`collect_eval_results_bonn.val_split_counts`), reads the two means from the
`metrics.csv` written by *this* run, and stores everything in the result
JSON. `val/all_psnr` (a pooled-MSE PSNR over all pixels of all views) is
recorded for reference but is **not** the table metric.

## 7. The table driver

```bash
DATA_ROOT="$BONN_VAL" OUTPUT_ROOT="$OUTPUT_ROOT" \
CKPT_ROOT="$CKPTS/checkpoints/stage2/Bonn" bash scripts/comparisons/run_table_bonn.sh

# one cell
MATERIALS=318 MODELS=Ours DATA_ROOT="$BONN_VAL" OUTPUT_ROOT="$OUTPUT_ROOT" \
CKPT_ROOT="$CKPTS/checkpoints/stage2/Bonn" bash scripts/comparisons/run_table_bonn.sh
```

The driver has the same fail-closed design as `scripts/run_table_ours.sh`
and `scripts/comparisons/run_table_ubo.sh` (described in
[reproduce_paper.md](reproduce_paper.md#running-the-table-drivers)):

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | required | the `Bonn_val` folder of §2 with `bonn_point_metadata.json` (§3). Checked up front: the metadata file, an entry for every requested material and the five per-material files must exist, otherwise exit 2 with every problem listed. |
| `CKPT_ROOT` | required | `$CKPTS/checkpoints/stage2/Bonn`; layout `<mat>/{Ours,Bonn,MERL,PBR}_epoch<N>.ckpt`, exactly one file per cell. |
| `OUTPUT_ROOT` | `outputs/full_experiments` under the repository | results `$OUTPUT_ROOT/eval_results_bonn/<mat>_<model>.json`; one experiment folder per cell, `Eval_Bonn_<mat>_<model>/`, with the CSV logger's `metrics.csv` (`Eval_Bonn_<mat>_<model>/version_<N>/`) and the saved views (`images/gt_mat<id>_view<i>_{poly,gray}.png`, `images/pred_mat<id>_view<i>_{poly,gray}_psnr<PSNR>.png`). |
| `MATERIALS`, `MODELS` | `318 377 32 226 37` × `Ours Bonn MERL PBR` | space-separated subsets; only materials/models with a paper value are accepted. |
| `TOLERANCE_DB` | `0.05` | maximum accepted `\|repro − paper\|` per cell; the average row gets 0.01 dB extra slack. |
| `FORCE` | `0` | `1` re-evaluates every cell. |
| `ALLOW_MISSING` | `0` | `1` reports missing/ambiguous checkpoints, evaluates the rest; those cells print as `MISSING` and the run still exits non-zero. |
| `SAVE_ALL_VIEWS` | `0` | `1` saves PNGs for all 133 held-out views instead of the first 20 (`data.valid_num=-1`). |
| `LLS_SPP` | unset | passes `model.lls_spp=<n>` to `train.py` (the shipped `stage2_bonn.yaml` resolves to 16). Recorded as an override in the result JSON, so results made with a different value are never reused. |
| `CKPT_SHA256` | `0` | `1` also records and compares the checkpoint's sha256. |
| `ROBOCLOTH_PYTHON` | `python` | interpreter for `train.py` and the collector (the generic `PYTHON` variable is ignored). |

What one run does: (1) resolve every checkpoint of the requested cells
**before** evaluating anything — missing or ambiguous → exit 4; (2) per
cell, reuse `<mat>_<model>.json` only if `collect_eval_results_bonn.py
--reuse-check` accepts it (it must record a `val_psnr`, this cell's
`material`, `model`, `experiment` = `stage2_bonn` / `stage2_bonn_pbr`,
`dataset_folder` (by real path) and `overrides`, and a checkpoint
path/size/mtime that still matches the file), otherwise set the old JSON
aside as `.json.prev`, evaluate, and accept only a JSON that describes this
checkpoint — anything else stops the run with exit 5 and prints the
single-cell command to re-run; (3) collect: print the table, exit 6 if a
cell is missing, 7 if one is beyond tolerance, 0 only when every expected
cell passes. A run can be interrupted and restarted; finished cells are
reused.

| Exit | Meaning |
|---|---|
| 0 | PASS — every expected cell within `TOLERANCE_DB` |
| 2 | bad environment: interpreter not executable, `DATA_ROOT` not a directory or not a usable Bonn folder (see above), unknown material/model |
| 4 | a checkpoint is missing or ambiguous (all cells checked first, all offenders listed) |
| 5 | an evaluation failed (`train.py` non-zero; no new `metrics.csv` written by this run; `val/poly_psnr` or `val/gray_psnr` missing) or left no valid JSON for its checkpoint |
| 6 | an expected table cell has no result |
| 7 | a cell differs from the paper by more than `TOLERANCE_DB` (takes precedence over 6) |

**The result JSON** (`collect_eval_results.write_result`, written atomically)
carries the shared provenance fields — `ckpt`, `ckpt_abs`, `ckpt_size`,
`ckpt_mtime_ns`, `ckpt_mtime` (+ `ckpt_sha256`), `git_head`, `git_dirty`,
`timestamp`, `hostname`, `provenance_version` — the identifiers `material`,
`model`, `experiment`, `dataset_folder`, `exp_name`, `valid_num`,
`metrics_csv`, `overrides`, and the Bonn metric: `val_psnr` (the table
number), `val_poly_psnr`, `val_gray_psnr`, `n_val_poly` (27), `n_val_gray`
(106), `n_val_views` (133), `val_all_psnr`, `val_poly_loss`, `val_gray_loss`
(`val_loss` is `null`: the trainer logs no single validation loss). The file
records your machine name and absolute paths.

A passing single-cell run ends like this (paths shortened):

```
[check-data] .../Bonn_val: bonn_point_metadata.json and all per-material files present for 318
[run_table_bonn] evaluating material 318 / Ours: no previous result (.../eval_results_bonn/318_Ours.json)
[run_table_bonn] evaluating material 318 / Ours (.../checkpoints/stage2/Bonn/318/Ours_epoch120.ckpt)
...
[run_table_bonn] material 318 / Ours: val psnr = 44.38 dB (poly 27 views: ..., gray 106 views: ...)  ->  .../eval_results_bonn/318_Ours.json

=== Per-material reconstruction PSNR (dB) — held-out Bonn (UBOFAB19) test set ===
results: .../eval_results_bonn
tolerance: |repro - paper| <= 0.05 dB per cell (average row: <= 0.06 dB, incl. rounding slack of the paper's 2-decimal averages)
material |                      Ours |                      Bonn |                      MERL |                       PBR
         |  repro/paper/diff  status |  repro/paper/diff  status |  repro/paper/diff  status |  repro/paper/diff  status
------------------------------------------------------------------------------------------------------------------------
     318 | 44.38/44.38/+0.00    PASS |    --/45.83/   --     n/a |    --/40.26/   --     n/a |    --/43.16/   --     n/a
     ...
 average |    --/23.43/   --   (1/5) |    --/24.47/   --   (0/5) |    --/21.77/   --   (0/5) |    --/22.99/   --   (0/5)

cells: 1 expected, 1 present, 0 missing; 1 PASS, 0 FAIL (tolerance 0.05 dB)
RESULT: PASS — all expected cells within tolerance
[run_table_bonn] PASS: all 1 cells reproduced within 0.05 dB of the paper
```

A second invocation prints `[run_table_bonn] reuse .../318_Ours.json:
checkpoint unchanged (path/size/mtime match); ...` and goes straight to the
table.

## 8. Expected metrics

Table 3, bottom block (PSNR in dB over the 133 held-out views of each
material; columns = the dataset the stage-1 decoder was trained on, plus the
Disney-PBR baseline). The same values are hard-coded in
`scripts/comparisons/collect_eval_results_bonn.py` (`PAPER`, `PAPER_AVG`).

| Material | RoboCloth (`Ours`) | Bonn | MERL | PBR |
|---|---|---|---|---|
| 318 | 44.38 | 45.83 | 40.26 | 43.16 |
| 377 | 16.29 | 16.78 | 15.57 | 15.93 |
| 32 | 15.31 | 16.42 | 14.18 | 15.17 |
| 226 | 24.14 | 25.09 | 22.46 | 23.92 |
| 37 | 17.03 | 18.22 | 16.36 | 16.75 |
| **Average** | **23.43** | **24.47** | **21.77** | **22.99** |

Notes on reproducing them:

* The paper values are the image-count-weighted `val/poly_psnr` /
  `val/gray_psnr` of the released checkpoints' final validation (§6); the
  driver re-runs exactly that validation step.
* LLS views are Monte-Carlo integrated; a fresh run draws different samples
  than the training-time validation did, so small noise-level deviations on
  those views are possible. The 377 checkpoints were logged with
  `model.lls_spp: 4` (their stored hyper-parameters) while the shipped config
  uses 16 — `LLS_SPP=4` reproduces the logged setting for that row.
* `TOLERANCE_DB` (default 0.05 dB, the margin observed when reproducing the
  RoboCloth block) is the pass criterion; the driver never rounds.
* Status: no Bonn cell has been re-run through this driver yet (see the
  [verification status](reproduce_paper.md#verification-status-of-this-repository)).
  What has been checked: a released Bonn checkpoint loads strictly into the
  current model built from `stage2_bonn.yaml` (all 13 tensors, CPU), the
  held-out split counts of all five materials (27 poly + 106 gray) from the
  released files, and the driver's fail-closed logic (§9).

## 9. Tests

The driver, the collector, the data check, the split counts and the weighted
metric are unit-tested without a GPU, Bonn data or checkpoints (stub
evaluator + a fake `python` that emulates `train.py`'s CSV logger; tiny EXR
files with the real channel naming stand in for the measurements):

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests/bonn_driver -v
```

The EXR-based tests need `pyexr` (part of the training environment) and are
skipped otherwise. Scratch files go to `ROBOCLOTH_CI_TMP` if set, else the
system temp directory.

## 10. Rendering Bonn checkpoints

The stage-2 Bonn checkpoints render with the Mitsuba plugin like any other
(`"material_type": "BonnLatentBRDF"` / `"BonnPBRLatentBRDF"` +
`"apply_cosine_at_eval": true` in `materials.json`); the wrapper reads the
per-material latent-grid shape from `bonn_point_metadata.json`, so point
`bonn_dataset_folder` in `rendering/configs/material/BonnLatentBRDF.yaml`
(and `BonnPBRLatentBRDF.yaml`) at your `Bonn_val` folder. See the Bonn
paragraph of [reproduce_paper.md](reproduce_paper.md#qualitative-figures-renders-of-the-checkpoints).
