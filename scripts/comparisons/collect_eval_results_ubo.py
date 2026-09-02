#!/usr/bin/env python3
"""Collect eval_stage2_ubo.sh results and check them against the paper table.

Usage:
  python collect_eval_results_ubo.py <eval_results_ubo_dir> [--materials "felt01 felt03"]
      [--models "Bonn"] [--tolerance-db 0.05] [--missing "felt01/Bonn ..."]
  python collect_eval_results_ubo.py --reuse-check <result.json> <checkpoint>

Prints the reproduced "Cross-dataset transfer to UBO2014" table (12 held-out
materials) next to the values reported in the paper with a PASS/FAIL verdict
per cell.  Same fail-closed contract and exit codes as scripts/collect_eval_results.py
(6 = expected cell missing, 7 = beyond tolerance, 2 = bad arguments), which
implements the table, the result-JSON schema and the --reuse-check.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from collect_eval_results import Table, main  # noqa: E402  (scripts/collect_eval_results.py)

# Paper: Table "Cross-dataset transfer to UBO2014" (12 held-out materials).
# Columns: stage-1 decoder training source (+ Disney-PBR baseline).
PAPER = {
    "fabric02": {"Ours": 34.57, "Bonn": 28.56, "MERL": 27.88, "PBR": 30.96},
    "fabric04": {"Ours": 34.00, "Bonn": 28.50, "MERL": 26.66, "PBR": 30.09},
    "fabric09": {"Ours": 39.50, "Bonn": 36.08, "MERL": 33.96, "PBR": 36.96},
    "fabric11": {"Ours": 34.33, "Bonn": 29.45, "MERL": 27.46, "PBR": 31.24},
    "felt01":   {"Ours": 38.76, "Bonn": 33.60, "MERL": 32.81, "PBR": 35.05},
    "felt03":   {"Ours": 40.30, "Bonn": 34.20, "MERL": 33.27, "PBR": 36.66},
    "felt05":   {"Ours": 35.39, "Bonn": 32.13, "MERL": 30.32, "PBR": 32.63},
    "felt10":   {"Ours": 44.36, "Bonn": 38.52, "MERL": 36.38, "PBR": 41.10},
    "carpet02": {"Ours": 38.87, "Bonn": 34.82, "MERL": 32.56, "PBR": 33.75},
    "carpet07": {"Ours": 34.61, "Bonn": 30.17, "MERL": 28.85, "PBR": 29.48},
    "carpet09": {"Ours": 33.17, "Bonn": 31.23, "MERL": 29.42, "PBR": 29.97},
    "carpet12": {"Ours": 33.25, "Bonn": 29.93, "MERL": 28.13, "PBR": 28.70},
}
PAPER_AVG = {"Ours": 36.76, "Bonn": 32.27, "MERL": 30.64, "PBR": 33.05}
MODELS = ["Ours", "Bonn", "MERL", "PBR"]

UBO = Table("Cross-dataset transfer to UBO2014 — PSNR (dB)", PAPER, PAPER_AVG,
            name_width=9, default_dir="eval_results_ubo")


if __name__ == "__main__":
    sys.exit(main(table=UBO))
