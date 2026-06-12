# -*- coding: utf-8 -*-
"""
ECMF-4DD multi-split launcher.

Edit only the configuration block near the top of this file for normal use.
The script repeatedly calls run_ecmf_4dd.py and aggregates the results.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# ============================================================
# Experiment plan
# ============================================================
# Common RUN_PRESET values:
#   "single_debug"       : run one split for a quick check
#   "ratio_7_1_2"        : run both datasets with the 7:1:2 split and five seeds
#   "mimic3_ratio_7_1_2" : run only the MIMIC-III-based dataset with five seeds
#   "mimic4_ratio_7_1_2" : run only the MIMIC-IV-based dataset with five seeds
#   "demo"               : run the bundled synthetic demo dataset
#   "custom"             : use CUSTOM_DATASETS / CUSTOM_RATIOS / CUSTOM_SEEDS
RUN_PRESET = "demo"

CUSTOM_DATASETS = "mimic3,mimic4"  # options: mimic3, mimic4, data3, data4, demo
CUSTOM_RATIOS = "7_1_2"
CUSTOM_SEEDS = "1,10,100,1000,10000"

# The following values are used by single_debug.
# For a quick real-data check, mimic4 is usually the harder dataset.
SINGLE_DATASET = "mimic4"
SINGLE_RATIO = "7_1_2"
SINGLE_SEED = "1"

# ============================================================
# ECMF-4DD training configuration
# ============================================================
# Model summary:
#   ECMF-4DD = Edge-aware Clinical Multi-view Fusion for Disease Diagnosis.
#   This implementation uses edge-aware patient-centered clinical views:
#     1. Patient-specific hop1 clinical subgraph;
#     2. Drug / Lab_Item / Procedure clinical views;
#     3. concept-level attention inside each view;
#     4. attention, message, and gate computation use edge_attr;
#     5. degree / IDF-aware denoising gate;
#     6. self/drug/lab/procedure view-level attention.


DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 30
DEFAULT_EARLY_STOP_MIN_DELTA = 1e-4
DEFAULT_EARLY_STOP_METRIC = "macro_f1"  # options: macro_f1 / macro_auprc

DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.5
DEFAULT_ATTN_DROPOUT = 0.10
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_USE_COUNT_FEATURE = 1
DEFAULT_USE_EDGE_ATTR = 1
DEFAULT_USE_DENOISE_GATE = 1
DEFAULT_IDF_BIAS_SCALE = 0.10

# ============================================================
# v1.5 semantic composite view switches
# ============================================================
# USE_SEMANTIC_COMPOSITE = 1 adds the following views:
#   evidence-view        = Lab-based evidence view
#   treatment-event-view = Drug + Procedure treatment-event view
#   complement-view      = interaction between evidence and treatment-event views
# These views do not introduce new neighbors or P-I-P metapaths.
USE_SEMANTIC_COMPOSITE = 1
USE_COMPLEMENT_VIEW = 1

DEFAULT_RUN_SEED = 2026

# Loss configuration:
#   "ce"           : standard cross entropy
#   "balanced_ce"  : class-weighted cross entropy
LOSS_MODE = "ce"
CLASS_WEIGHT_POWER = 0.5
LABEL_SMOOTHING = 0.0

# ============================================================
# Batch-size configuration
# ============================================================
# BATCH_SIZE:
#   Number of target Patient nodes per training batch.
#   All one-hop item neighbors are kept for each Patient.
#
# EVAL_BATCH_SIZE:
#   Number of target Patient nodes per validation/test batch.
#   Evaluation does not backpropagate, so this can be larger.
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 2048

# ============================================================
# Logging configuration
# ============================================================
# STREAM_CHILD_OUTPUT=True streams child-process logs to the terminal.
# If False, logs are written only to files.
STREAM_CHILD_OUTPUT = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = BASELINE_DIR / "run_ecmf_4dd.py"
RESULT_ROOT = PROJECT_ROOT / "results" / "ecmf_4dd"
AGG_ROOT = RESULT_ROOT / "_aggregate"


def split_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def build_split_name(ratio: str, seed: str) -> str:
    return f"ratio_{ratio}_seed_{seed}"


def resolve_plan() -> tuple[List[str], List[str], List[str]]:
    if RUN_PRESET == "single_debug":
        return [SINGLE_DATASET], [SINGLE_RATIO], [SINGLE_SEED]
    if RUN_PRESET == "demo":
        return ["demo"], ["7_1_2"], ["1"]
    if RUN_PRESET == "ratio_7_1_2":
        return ["mimic3", "mimic4"], ["7_1_2"], ["1", "10", "100", "1000", "10000"]
    if RUN_PRESET == "ratio_8_1_1":
        return ["mimic3", "mimic4"], ["8_1_1"], ["1", "10", "100", "1000", "10000"]
    if RUN_PRESET == "ratio_6_2_2":
        return ["mimic3", "mimic4"], ["6_2_2"], ["1", "10", "100", "1000", "10000"]
    if RUN_PRESET == "mimic3_ratio_7_1_2":
        return ["mimic3"], ["7_1_2"], ["1", "10", "100", "1000", "10000"]
    if RUN_PRESET == "mimic4_ratio_7_1_2":
        return ["mimic4"], ["7_1_2"], ["1", "10", "100", "1000", "10000"]
    if RUN_PRESET == "all_30":
        return ["mimic3", "mimic4"], ["8_1_1", "7_1_2", "6_2_2"], ["1", "10", "100", "1000", "10000"]
    if RUN_PRESET == "custom":
        return split_csv(CUSTOM_DATASETS), split_csv(CUSTOM_RATIOS), split_csv(CUSTOM_SEEDS)
    raise ValueError(f"Unknown RUN_PRESET={RUN_PRESET}")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def run_child(cmd: List[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8", newline="") as log_f:
        if STREAM_CHILD_OUTPUT:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log_f.write(line)
            return int(proc.wait())
        completed = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log_f.write(completed.stdout)
        return int(completed.returncode)


def aggregate_rows(run_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    ok_rows = [r for r in run_rows if r.get("status") == "success"]
    detail_rows: List[Dict[str, Any]] = list(ok_rows)
    paper_rows: List[Dict[str, Any]] = []

    for dataset in sorted({r["dataset"] for r in ok_rows}):
        ds_rows = [r for r in ok_rows if r["dataset"] == dataset]
        for ratio in sorted({r["ratio"] for r in ds_rows}):
            group = [r for r in ds_rows if r["ratio"] == ratio]
            if not group:
                continue

            def vals(key: str) -> List[float]:
                return [float(x[key]) for x in group]

            import statistics

            def mean_std(key: str) -> tuple[float, float]:
                xs = vals(key)
                if len(xs) <= 1:
                    return xs[0], 0.0
                return statistics.mean(xs), statistics.stdev(xs)

            micro_m, micro_s = mean_std("test_micro_f1")
            macro_m, macro_s = mean_std("test_macro_f1")
            auprc_m, auprc_s = mean_std("test_macro_auprc")
            auroc_m, auroc_s = mean_std("test_macro_auroc")
            gpu_m, gpu_s = mean_std("cuda_peak_mb")
            time_m, time_s = mean_std("time_sec")
            paper_rows.append({
                "Model": "ECMF-4DD-v1",
                "Dataset": dataset,
                "Split Group": f"ratio_{ratio}",
                "#Splits": len(group),
                "Micro-F1": format_metric(micro_m, micro_s),
                "Macro-F1": format_metric(macro_m, macro_s),
                "Macro-AUPRC": format_metric(auprc_m, auprc_s),
                "Macro-AUROC": format_metric(auroc_m, auroc_s),
                "GPU MB": format_metric(gpu_m, gpu_s),
                "Train Sec": format_metric(time_m, time_s),
                "Notes": "ECMF-4DD; edge-aware patient-specific hop1 clinical subgraph; edge_attr + degree/IDF denoising gate; no global branch.",
            })
    write_csv(out_dir / "ecmf_4dd_30splits_detail.csv", detail_rows)
    write_csv(out_dir / "ecmf_4dd_paper_table_ready.csv", paper_rows)
    write_json(out_dir / "ecmf_4dd_runner_summary.json", {
        "num_runs": len(run_rows),
        "num_success": len(ok_rows),
        "num_failed": len(run_rows) - len(ok_rows),
        "run_preset": RUN_PRESET,
    })


def main() -> None:
    datasets, ratios, seeds = resolve_plan()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = AGG_ROOT / f"{ts}_ecmf_4dd_splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("ECMF-4DD runner")
    print(f"RUN_PRESET={RUN_PRESET}")
    print(f"datasets={datasets}, ratios={ratios}, seeds={seeds}")
    print(f"out_dir={out_dir}")
    print("=" * 80)

    run_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        for ratio in ratios:
            for seed in seeds:
                split = build_split_name(ratio, seed)
                run_name = f"{dataset}_{split}"
                result_dir = RESULT_ROOT / f"{ts}_{run_name}"
                log_path = out_dir / f"log_{run_name}.txt"
                cmd = [
                    sys.executable,
                    str(RUN_SCRIPT),
                    "--dataset", dataset,
                    "--split", split,
                    "--epochs", str(DEFAULT_EPOCHS),
                    "--patience", str(DEFAULT_PATIENCE),
                    "--early-stop-min-delta", str(DEFAULT_EARLY_STOP_MIN_DELTA),
                    "--early-stop-metric", str(DEFAULT_EARLY_STOP_METRIC),
                    "--hidden-dim", str(DEFAULT_HIDDEN_DIM),
                    "--dropout", str(DEFAULT_DROPOUT),
                    "--attn-dropout", str(DEFAULT_ATTN_DROPOUT),
                    "--lr", str(DEFAULT_LR),
                    "--weight-decay", str(DEFAULT_WEIGHT_DECAY),
                    "--grad-clip", str(DEFAULT_GRAD_CLIP),
                    "--batch-size", str(BATCH_SIZE),
                    "--eval-batch-size", str(EVAL_BATCH_SIZE),
                    "--use-count-feature", str(DEFAULT_USE_COUNT_FEATURE),
                    "--use-edge-attr", str(DEFAULT_USE_EDGE_ATTR),
                    "--use-denoise-gate", str(DEFAULT_USE_DENOISE_GATE),
                    "--idf-bias-scale", str(DEFAULT_IDF_BIAS_SCALE),
                    "--use-semantic-composite", str(USE_SEMANTIC_COMPOSITE),
                    "--use-complement-view", str(USE_COMPLEMENT_VIEW),
                    "--loss-mode", str(LOSS_MODE),
                    "--class-weight-power", str(CLASS_WEIGHT_POWER),
                    "--label-smoothing", str(LABEL_SMOOTHING),
                    "--run-seed", str(DEFAULT_RUN_SEED),
                    "--result-dir", str(result_dir),
                ]
                print("-" * 80)
                print(f"[ECMF-4DD-runner] START {run_name}")
                print(f"[ECMF-4DD-runner] cmd={' '.join(cmd)}")
                start = time.time()
                code = run_child(cmd, log_path)
                elapsed = time.time() - start
                row: Dict[str, Any] = {
                    "dataset": dataset,
                    "ratio": ratio,
                    "seed": seed,
                    "split": split,
                    "result_dir": str(result_dir),
                    "log_path": str(log_path),
                    "return_code": code,
                    "runner_time_sec": elapsed,
                }
                if code == 0 and (result_dir / "results_summary.json").exists():
                    summary = read_json(result_dir / "results_summary.json")
                    row.update(summary)
                    row["status"] = "success"
                    print(f"[ECMF-4DD-runner][OK] {run_name}")
                else:
                    row["status"] = "failed"
                    print(f"[ECMF-4DD-runner][FAILED] {run_name}, return_code={code}, log={log_path}")
                run_rows.append(row)
                write_csv(out_dir / "ecmf_4dd_running_detail.csv", run_rows)
    aggregate_rows(run_rows, out_dir)
    print("=" * 80)
    print(f"[ECMF-4DD-runner] DONE. Summary: {out_dir / 'ecmf_4dd_paper_table_ready.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
