# -*- coding: utf-8 -*-
"""
Inspect edge-attribute statistics for a GNN-ready dataset.

This utility checks edge_attr dimensions, missing values, all-zero ratios,
and simple summary statistics before training ECMF-4DD.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def load_graph(dataset: str, split: str = "ratio_7_1_2_seed_1"):
    from data_loading.gnn_ready_dataset_loader import GNNReadyDatasetConfig, load_gnn_ready_dataset

    cfg = GNNReadyDatasetConfig(
        project_root=PROJECT_ROOT,
        dataset_key=dataset,
        split_name=split,
        backend="dgl",
        device="cpu",
        verbose=True,
    )
    loaded = load_gnn_ready_dataset(cfg)
    return loaded.graph, loaded


def summarize_tensor(x: torch.Tensor) -> Dict[str, Any]:
    x = x.detach().cpu().float()
    out: Dict[str, Any] = {
        "shape": str(tuple(x.shape)),
        "numel": int(x.numel()),
    }
    if x.numel() == 0:
        out.update({"nan_count": 0, "all_zero_row_rate": "", "mean": "", "std": "", "min": "", "max": ""})
        return out
    nan_mask = torch.isnan(x)
    out["nan_count"] = int(nan_mask.sum().item())
    x_safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x_safe.ndim == 2:
        out["all_zero_row_rate"] = float((x_safe.abs().sum(dim=1) == 0).float().mean().item())
    else:
        out["all_zero_row_rate"] = ""
    out["mean"] = float(x_safe.mean().item())
    out["std"] = float(x_safe.std().item()) if x_safe.numel() > 1 else 0.0
    out["min"] = float(x_safe.min().item())
    out["max"] = float(x_safe.max().item())
    return out


def main() -> None:
    rows: List[Dict[str, Any]] = []
    for dataset in ["mimic3", "mimic4"]:
        g, loaded = load_graph(dataset)
        for etype in g.canonical_etypes:
            e_data = g.edges[etype].data
            row: Dict[str, Any] = {
                "dataset": dataset,
                "dataset_name": loaded.dataset_name,
                "split": loaded.split_name,
                "canonical_etype": str(etype),
                "num_edges": int(g.num_edges(etype)),
                "has_edge_attr": "edge_attr" in e_data,
            }
            if "edge_attr" in e_data:
                row.update(summarize_tensor(e_data["edge_attr"]))
            rows.append(row)
            print(row)
    out_path = PROJECT_ROOT / "results" / "ecmf_4dd" / "edge_attr_inspection.csv"
    write_csv_rows(out_path, rows)
    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()
