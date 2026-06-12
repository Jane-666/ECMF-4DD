# -*- coding: utf-8 -*-
"""Regenerate the tiny synthetic demo dataset.

The demo data are random tensors for smoke tests only. They contain no real
patient information and are not used for paper results.
"""

from pathlib import Path
import json
import torch


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    split_dir = repo_root / "final_datasets" / "gnn_ready" / "demo" / "ratio_7_1_2_seed_1"
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    num_patient, num_drug, num_lab, num_proc, num_classes = 60, 15, 12, 10, 6
    features = {
        "Patient": torch.randn(num_patient, 85),
        "Drug": torch.randn(num_drug, 131),
        "Lab_Item": torch.randn(num_lab, 133),
        "Procedure": torch.randn(num_proc, 131),
    }
    y = torch.arange(num_patient) % num_classes
    perm = torch.randperm(num_patient)
    train_mask = torch.zeros(num_patient, dtype=torch.bool)
    val_mask = torch.zeros(num_patient, dtype=torch.bool)
    test_mask = torch.zeros(num_patient, dtype=torch.bool)
    train_mask[perm[:42]] = True
    val_mask[perm[42:48]] = True
    test_mask[perm[48:]] = True

    def make_edges(n_item: int, per_patient: int, edge_dim: int, offset: int):
        src, dst = [], []
        for p in range(num_patient):
            for k in range(per_patient):
                src.append(p)
                dst.append((p * 3 + k + offset) % n_item)
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        edge_attr = torch.randn(len(src), edge_dim)
        return src, dst, edge_attr

    d_src, d_dst, d_attr = make_edges(num_drug, 3, 8, 0)
    l_src, l_dst, l_attr = make_edges(num_lab, 4, 10, 1)
    p_src, p_dst, p_attr = make_edges(num_proc, 2, 6, 2)
    raw_pkg = {
        "node_features": {k: {"x": v} for k, v in features.items()},
        "edges": {
            "patient_drug": {"canonical_etype": ["Patient", "uses_drug", "Drug"], "src": d_src, "dst": d_dst, "edge_attr": d_attr},
            "patient_lab": {"canonical_etype": ["Patient", "has_lab", "Lab_Item"], "src": l_src, "dst": l_dst, "edge_attr": l_attr},
            "patient_procedure": {"canonical_etype": ["Patient", "has_procedure", "Procedure"], "src": p_src, "dst": p_dst, "edge_attr": p_attr},
        },
        "supervision": {"y": y.long(), "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask},
    }
    torch.save(raw_pkg, split_dir / "graph_raw_tensors.pt")
    schema = {
        "dataset": "synthetic_demo",
        "warning": "Synthetic demo data only. Not used in paper experiments.",
        "node_summary": {
            "Patient": {"num_nodes": num_patient, "feature_dim": 85},
            "Drug": {"num_nodes": num_drug, "feature_dim": 131},
            "Lab_Item": {"num_nodes": num_lab, "feature_dim": 133},
            "Procedure": {"num_nodes": num_proc, "feature_dim": 131},
        },
        "edge_summary": {
            "patient_drug": {"canonical_etype": ["Patient", "uses_drug", "Drug"], "num_edges": int(d_src.numel()), "edge_attr_dim": 8},
            "patient_lab": {"canonical_etype": ["Patient", "has_lab", "Lab_Item"], "num_edges": int(l_src.numel()), "edge_attr_dim": 10},
            "patient_procedure": {"canonical_etype": ["Patient", "has_procedure", "Procedure"], "num_edges": int(p_src.numel()), "edge_attr_dim": 6},
        },
        "supervision_summary": {"num_classes": num_classes, "train": int(train_mask.sum()), "val": int(val_mask.sum()), "test": int(test_mask.sum())},
    }
    (split_dir / "graph_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    registry = split_dir.parent / "gnn_dataset_registry.csv"
    registry.write_text("SPLIT_NAME,RATIO,SEED,NOTE\nratio_7_1_2_seed_1,7_1_2,1,synthetic_demo_for_smoke_test_only\n", encoding="utf-8")
    print(f"Demo dataset written to: {split_dir.parent}")


if __name__ == "__main__":
    main()
