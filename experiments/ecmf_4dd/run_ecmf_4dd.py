# -*- coding: utf-8 -*-
"""
ECMF-4DD training script.

This file keeps the original ECRD-Net v1.5 model logic used in the paper,
while exposing it under the final paper name ECMF-4DD.

Main design:
- Patient-centered one-hop clinical neighborhoods.
- Drug, Lab_Item, and Procedure relation branches.
- Edge-aware scoring and message generation.
- Degree/IDF-aware relation denoising.
- Evidence, treatment-event, and complement semantic views.
- Patient-level disease diagnosis.

The training hyperparameters and model computation are kept unchanged from the
submitted ECMF-4DD experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ECMF-4DD requires DGL.") from exc

try:
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except Exception as exc:  # pragma: no cover
    raise RuntimeError("ECMF-4DD requires scikit-learn for metrics.") from exc

# ============================================================
# Warning filtering
# ============================================================
# Suppress a known DGL CPU-affinity warning that does not affect training.

try:
    from dgl.base import DGLWarning

    warnings.filterwarnings(
        "ignore",
        category=DGLWarning,
        message=".*Dataloader CPU affinity opt is not enabled.*",
    )
except Exception:
    warnings.filterwarnings(
        "ignore",
        message=".*Dataloader CPU affinity opt is not enabled.*",
    )

# ============================================================
# Repository path setup for both right-click and command-line execution
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Basic utility functions
# ============================================================
def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=True, indent=2)


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: to_jsonable(r.get(k, "")) for k in fieldnames})


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def _mask_to_idx(mask: Any, expected_len: Optional[int] = None, name: str = "mask") -> torch.Tensor:
    """Convert a boolean mask, 0/1 mask, or index list to a LongTensor index."""
    t = torch.as_tensor(mask).detach().cpu()
    if t.ndim > 1:
        t = t.view(-1)
    if t.dtype == torch.bool:
        return torch.nonzero(t, as_tuple=False).view(-1).long()
    if expected_len is not None and int(t.numel()) == int(expected_len):
        uniq = torch.unique(t)
        if uniq.numel() <= 2 and all(float(x.item()) in (0.0, 1.0) for x in uniq):
            return torch.nonzero(t.to(torch.bool), as_tuple=False).view(-1).long()
    idx = t.long().view(-1)
    if expected_len is not None and idx.numel() > 0:
        if int(idx.min().item()) < 0 or int(idx.max().item()) >= int(expected_len):
            raise ValueError(f"{name} index out of range: min={int(idx.min())}, max={int(idx.max())}, expected_len={expected_len}")
    if expected_len is not None and idx.numel() > 8:
        uniq = torch.unique(idx)
        if uniq.numel() <= 2 and all(int(x.item()) in (0, 1) for x in uniq):
            raise ValueError(f"{name} looks like a 0/1 mask but was not parsed correctly.")
    return idx


def _normalize_labels(labels: Any, expected_len: Optional[int] = None) -> torch.Tensor:
    y = torch.as_tensor(labels).detach().cpu()
    if y.ndim == 2:
        if y.shape[1] == 1:
            y = y.view(-1)
        else:
            y = y.argmax(dim=1)
    else:
        y = y.view(-1)
    y = y.long()
    if expected_len is not None and int(y.numel()) != int(expected_len):
        raise ValueError(f"label length does not match the number of Patient nodes: labels={y.numel()}, Patient={expected_len}")
    return y


# ============================================================
# Dataset loading through the unified GNN-ready loader
# ============================================================
@dataclass
class LoadedSplit:
    hg: Any
    node_features: Dict[str, torch.Tensor]
    labels: torch.Tensor
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    num_classes: int
    dataset_name: str
    split_name: str


def load_split(dataset: str, split: str, project_root: Path) -> LoadedSplit:
    from data_loading.gnn_ready_dataset_loader import GNNReadyDatasetConfig, load_gnn_ready_dataset

    cfg = GNNReadyDatasetConfig(
        project_root=project_root,
        dataset_key=dataset,
        split_name=split,
        backend="dgl",
        device="cpu",
        verbose=True,
    )
    loaded = load_gnn_ready_dataset(cfg)
    hg = loaded.graph
    if hg is None:
        raise RuntimeError("The unified loader did not return a DGL graph. Please check backend='dgl'.")
    if "Patient" not in hg.ntypes:
        raise RuntimeError(f"Patient node type is missing from the graph: ntypes={hg.ntypes}")
    pdata = hg.nodes["Patient"].data
    for key in ["x", "y", "train_mask", "val_mask", "test_mask"]:
        if key not in pdata:
            raise RuntimeError(f"Patient node data is missing field {key}. Existing fields: {list(pdata.keys())}")

    num_patient = int(hg.num_nodes("Patient"))
    labels = _normalize_labels(pdata["y"], expected_len=num_patient)
    train_idx = _mask_to_idx(pdata["train_mask"], expected_len=num_patient, name="train_mask")
    val_idx = _mask_to_idx(pdata["val_mask"], expected_len=num_patient, name="val_mask")
    test_idx = _mask_to_idx(pdata["test_mask"], expected_len=num_patient, name="test_mask")
    num_classes = int(labels.max().item()) + 1

    node_features: Dict[str, torch.Tensor] = {}
    for ntype in hg.ntypes:
        if "x" not in hg.nodes[ntype].data:
            raise RuntimeError(f"Node type {ntype} is missing x features.")
        node_features[ntype] = hg.nodes[ntype].data["x"].detach().cpu().float()

    def hist(idx: torch.Tensor) -> List[int]:
        return torch.bincount(labels[idx], minlength=num_classes).tolist()

    print("=" * 80)
    print("ECMF-4DD dataset loaded")
    print(f"dataset      : {dataset}")
    print(f"dataset_name : {loaded.dataset_name}")
    print(f"split_name   : {loaded.split_name}")
    print(f"node_types   : {list(hg.ntypes)}")
    print("canonical_etypes:")
    for et in hg.canonical_etypes:
        e_info = hg.edges[et].data
        edge_attr_shape = tuple(e_info["edge_attr"].shape) if "edge_attr" in e_info else None
        print(f"  - {et}, edges={hg.num_edges(et)}, edge_attr={edge_attr_shape}")
    print(f"labels       : {tuple(labels.shape)}")
    print(f"train/val/test: {train_idx.numel()}/{val_idx.numel()}/{test_idx.numel()}")
    print(f"label_hist train={hist(train_idx)}")
    print(f"label_hist val  ={hist(val_idx)}")
    print(f"label_hist test ={hist(test_idx)}")
    print("=" * 80)

    return LoadedSplit(
        hg=hg,
        node_features=node_features,
        labels=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_classes=num_classes,
        dataset_name=loaded.dataset_name,
        split_name=loaded.split_name,
    )


# ============================================================
# One-hop patient-specific clinical neighbor maps with edge attributes
# ============================================================
@dataclass
class EdgeNeighborMap:
    item_type: str
    canonical_etype: Tuple[str, str, str]
    item_ids: List[torch.Tensor]
    edge_ids: List[torch.Tensor]
    edge_attr: torch.Tensor
    edge_attr_dim: int
    item_degree: torch.Tensor
    item_idf: torch.Tensor


def find_patient_to_item_etype(hg: Any, item_type: str) -> Tuple[str, str, str]:
    """Find the canonical Patient-to-item edge type."""
    candidates = [et for et in hg.canonical_etypes if et[0] == "Patient" and et[2] == item_type]
    if not candidates:
        raise RuntimeError(f"No Patient -> {item_type} edge type was found. canonical_etypes={hg.canonical_etypes}")
    return candidates[0]


def build_edge_neighbor_map(hg: Any, item_type: str) -> EdgeNeighborMap:
    etype = find_patient_to_item_etype(hg, item_type)
    src, dst, eid = hg.edges(etype=etype, form="all")
    src = src.detach().cpu().long()
    dst = dst.detach().cpu().long()
    eid = eid.detach().cpu().long()
    num_patient = int(hg.num_nodes("Patient"))
    num_item = int(hg.num_nodes(item_type))

    if "edge_attr" in hg.edges[etype].data:
        edge_attr = hg.edges[etype].data["edge_attr"].detach().cpu().float()
        if edge_attr.ndim == 1:
            edge_attr = edge_attr.view(-1, 1)
    else:
        edge_attr = torch.zeros((int(hg.num_edges(etype)), 0), dtype=torch.float32)
    edge_attr_dim = int(edge_attr.shape[1])

    item_lists: List[List[int]] = [[] for _ in range(num_patient)]
    eid_lists: List[List[int]] = [[] for _ in range(num_patient)]
    for p, it, e in zip(src.tolist(), dst.tolist(), eid.tolist()):
        item_lists[int(p)].append(int(it))
        eid_lists[int(p)].append(int(e))

    item_ids: List[torch.Tensor] = []
    edge_ids: List[torch.Tensor] = []
    for xs, es in zip(item_lists, eid_lists):
        if xs:
            item_ids.append(torch.tensor(xs, dtype=torch.long))
            edge_ids.append(torch.tensor(es, dtype=torch.long))
        else:
            item_ids.append(torch.empty(0, dtype=torch.long))
            edge_ids.append(torch.empty(0, dtype=torch.long))

    item_degree = torch.bincount(dst, minlength=num_item).float()
    # IDF is smaller for more frequent clinical items.
    item_idf = torch.log((float(num_patient) + 1.0) / (item_degree + 1.0))

    degs = torch.tensor([x.numel() for x in item_ids], dtype=torch.float)
    if edge_attr_dim > 0:
        nan_count = int(torch.isnan(edge_attr).sum().item())
        zero_rate = float((edge_attr.abs().sum(dim=1) == 0).float().mean().item()) if edge_attr.numel() > 0 else 0.0
    else:
        nan_count = 0
        zero_rate = 1.0
    print(
        f"[ECMF-4DD-neighbor] Patient->{item_type}: mean={degs.mean().item():.2f}, "
        f"median={degs.median().item():.0f}, p90={torch.quantile(degs, 0.90).item():.0f}, "
        f"max={degs.max().item():.0f}, edge_attr_dim={edge_attr_dim}, nan={nan_count}, zero_edge_rate={zero_rate:.4f}"
    )
    print(
        f"[ECMF-4DD-item-degree] {item_type}: mean={item_degree.mean().item():.2f}, "
        f"p90={torch.quantile(item_degree, 0.90).item():.0f}, max={item_degree.max().item():.0f}"
    )
    return EdgeNeighborMap(
        item_type=item_type,
        canonical_etype=etype,
        item_ids=item_ids,
        edge_ids=edge_ids,
        edge_attr=edge_attr,
        edge_attr_dim=edge_attr_dim,
        item_degree=item_degree,
        item_idf=item_idf,
    )


# ============================================================
# Metric computation and result writing
# ============================================================
def compute_metrics(y_true: np.ndarray, logits: np.ndarray, num_classes: int) -> Dict[str, float]:
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()
    pred = probs.argmax(axis=1)
    out: Dict[str, float] = {}
    out["micro_f1"] = float(f1_score(y_true, pred, average="micro", zero_division=0))
    out["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    try:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        out["macro_auprc"] = float(average_precision_score(y_bin, probs, average="macro"))
    except Exception:
        out["macro_auprc"] = float("nan")
    try:
        out["macro_auroc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=list(range(num_classes))))
    except Exception:
        out["macro_auroc"] = float("nan")
    return out


def write_confusion_matrix(path: Path, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    rows: List[Dict[str, Any]] = []
    for i in range(num_classes):
        row: Dict[str, Any] = {"true_label": i}
        for j in range(num_classes):
            row[f"pred_{j}"] = int(cm[i, j])
        rows.append(row)
    write_csv_rows(path, rows)


def make_class_weight(labels: torch.Tensor, train_idx: torch.Tensor, num_classes: int, power: float = 0.5) -> torch.Tensor:
    y = labels[train_idx].detach().cpu().long()
    counts = torch.bincount(y, minlength=num_classes).float().clamp_min(1.0)
    inv = counts.sum() / counts
    weight = inv.pow(float(power))
    weight = weight / weight.mean().clamp_min(1e-12)
    return weight.float()


# ============================================================
# Model modules
# ============================================================
class TypeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeEncoder(nn.Module):
    """Encode relation-specific edge attributes into the hidden space.

    If a relation has no edge attributes, zero vectors are returned.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        if self.in_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.net = None

    def forward(self, edge_x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return edge_x.new_zeros((edge_x.shape[0], self.hidden_dim))
        return self.net(edge_x)


class ECRDNetV15(nn.Module):
    """Edge-aware patient-specific clinical subgraph model.

    Difference from a plain patient-view attention model:
    1. concept attention uses edge attributes;
    2. messages use item features and edge attributes;
    3. attention includes a degree/IDF-aware denoising gate.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        edge_dims: Dict[str, int],
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        attn_dropout: float,
        use_count_feature: bool = True,
        use_edge_attr: bool = True,
        use_denoise_gate: bool = True,
        idf_bias_scale: float = 0.10,
        use_semantic_composite: bool = True,
        use_complement_view: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.view_types = ["Drug", "Lab_Item", "Procedure"]
        self.use_count_feature = bool(use_count_feature)
        self.use_edge_attr = bool(use_edge_attr)
        self.use_denoise_gate = bool(use_denoise_gate)
        self.use_semantic_composite = bool(use_semantic_composite)
        self.use_complement_view = bool(use_complement_view)

        self.encoders = nn.ModuleDict({
            ntype: TypeEncoder(in_dim, hidden_dim, dropout)
            for ntype, in_dim in input_dims.items()
        })
        self.edge_encoders = nn.ModuleDict({
            vt: EdgeEncoder(edge_dims.get(vt, 0) if self.use_edge_attr else 0, hidden_dim, dropout)
            for vt in self.view_types
        })

        self.query_proj = nn.ModuleDict({vt: nn.Linear(hidden_dim, hidden_dim, bias=False) for vt in self.view_types})
        self.key_proj = nn.ModuleDict({vt: nn.Linear(hidden_dim, hidden_dim, bias=False) for vt in self.view_types})
        self.message_proj = nn.ModuleDict({
            vt: nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for vt in self.view_types
        })
        self.edge_score = nn.ModuleDict({
            vt: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1, bias=False),
            )
            for vt in self.view_types
        })
        # Gate input: item_h, edge_h, idf, and log-normalized degree.
        self.gate_mlp = nn.ModuleDict({
            vt: nn.Sequential(
                nn.Linear(hidden_dim * 2 + 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for vt in self.view_types
        })
        self.idf_scale = nn.ParameterDict({
            vt: nn.Parameter(torch.tensor(float(idf_bias_scale))) for vt in self.view_types
        })
        self.empty_view = nn.ParameterDict({vt: nn.Parameter(torch.zeros(hidden_dim)) for vt in self.view_types})

        self.view_type_emb = nn.ParameterDict({
            "self": nn.Parameter(torch.zeros(hidden_dim)),
            "Drug": nn.Parameter(torch.zeros(hidden_dim)),
            "Lab_Item": nn.Parameter(torch.zeros(hidden_dim)),
            "Procedure": nn.Parameter(torch.zeros(hidden_dim)),
            # v1.5 semantic composite view tokens:
            # evidence = Lab-based evidence view; treatment_event = Drug + Procedure view;
            # complement = interaction between evidence and treatment-event views.
            "evidence": nn.Parameter(torch.zeros(hidden_dim)),
            "treatment_event": nn.Parameter(torch.zeros(hidden_dim)),
            "complement": nn.Parameter(torch.zeros(hidden_dim)),
        })
        for p in self.view_type_emb.values():
            nn.init.normal_(p, std=0.02)

        if self.use_count_feature:
            self.count_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        else:
            self.count_encoder = None

        # ============================================================
        # v1.5 clinical semantic composite views
        # ============================================================
        # treatment_event: Drug + Procedure, representing treatment or procedure events.
        # evidence: Lab, representing laboratory evidence.
        # complement: interaction between evidence and treatment-event views.
        self.treatment_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.evidence_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.complement_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.view_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _segment_softmax(self, scores: torch.Tensor, group: torch.Tensor, num_groups: int) -> torch.Tensor:
        max_init = torch.full((num_groups,), -1e30, device=scores.device, dtype=scores.dtype)
        max_per_group = max_init.scatter_reduce(0, group, scores, reduce="amax", include_self=True)
        exp = torch.exp(scores - max_per_group[group])
        denom = torch.zeros((num_groups,), device=scores.device, dtype=scores.dtype)
        denom.index_add_(0, group, exp)
        return exp / denom[group].clamp_min(1e-12)

    def _aggregate_view(
        self,
        vt: str,
        h_patient: torch.Tensor,
        h_item_all: torch.Tensor,
        batch_patient_ids: torch.Tensor,
        neigh_map: EdgeNeighborMap,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = int(batch_patient_ids.numel())
        local_index: List[torch.Tensor] = []
        item_index: List[torch.Tensor] = []
        edge_index: List[torch.Tensor] = []
        counts = torch.zeros((bsz,), dtype=torch.float32, device=device)
        for i, p in enumerate(batch_patient_ids.detach().cpu().tolist()):
            item_ids = neigh_map.item_ids[int(p)]
            edge_ids = neigh_map.edge_ids[int(p)]
            n = int(item_ids.numel())
            counts[i] = float(n)
            if n > 0:
                local_index.append(torch.full((n,), i, dtype=torch.long))
                item_index.append(item_ids.long())
                edge_index.append(edge_ids.long())

        if not item_index:
            view = self.empty_view[vt].view(1, -1).expand(bsz, -1)
            if self.count_encoder is not None:
                view = view + self.count_encoder(torch.log1p(counts).view(-1, 1))
            return view, counts

        group = torch.cat(local_index, dim=0).to(device)
        item_ids = torch.cat(item_index, dim=0).to(device)
        edge_ids = torch.cat(edge_index, dim=0).to(device)

        item_h = h_item_all[item_ids]
        raw_edge_attr = neigh_map.edge_attr[edge_ids.detach().cpu()].to(device, non_blocking=True)
        if not self.use_edge_attr:
            raw_edge_attr = raw_edge_attr.new_zeros((raw_edge_attr.shape[0], 0))
        edge_h = self.edge_encoders[vt](raw_edge_attr)

        q = self.query_proj[vt](h_patient)[group]
        # Both keys and messages are conditioned on item and edge representations.
        item_edge = item_h + edge_h
        k = self.key_proj[vt](item_edge)
        msg = self.message_proj[vt](torch.cat([item_h, edge_h], dim=-1))

        scores = (q * k).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        scores = scores + self.edge_score[vt](edge_h).squeeze(-1)

        idf = neigh_map.item_idf[item_ids.detach().cpu()].to(device).float()
        deg = neigh_map.item_degree[item_ids.detach().cpu()].to(device).float()
        log_deg_norm = torch.log1p(deg) / math.log(float(neigh_map.item_degree.numel()) + float(deg.max().item()) + 2.0)
        scores = scores + self.idf_scale[vt] * idf

        if self.use_denoise_gate:
            gate_in = torch.cat([item_h, edge_h, idf.view(-1, 1), log_deg_norm.view(-1, 1)], dim=-1)
            gate = torch.sigmoid(self.gate_mlp[vt](gate_in).squeeze(-1)).clamp_min(1e-5)
            # log(gate) enters softmax as a learnable denoising bias.
            scores = scores + torch.log(gate)

        scores = F.leaky_relu(scores, negative_slope=0.2)
        alpha = self._segment_softmax(scores, group, bsz).unsqueeze(-1)
        alpha = self.attn_dropout(alpha)
        out = torch.zeros((bsz, self.hidden_dim), device=device, dtype=h_patient.dtype)
        out.index_add_(0, group, alpha * msg)

        empty_mask = counts <= 0
        if bool(empty_mask.any().item()):
            out[empty_mask] = self.empty_view[vt]
        if self.count_encoder is not None:
            out = out + self.count_encoder(torch.log1p(counts).view(-1, 1))
        return out, counts

    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        batch_patient_ids: torch.Tensor,
        neighbor_maps: Dict[str, EdgeNeighborMap],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Encode only target Patient nodes in the current mini-batch.
        batch_patient_ids_cpu = batch_patient_ids.detach().cpu().long()
        batch_patient_ids = batch_patient_ids_cpu.to(device)
        patient_x = node_features["Patient"][batch_patient_ids_cpu].to(device, non_blocking=True)
        h_patient = self.encoders["Patient"](patient_x)

        h_items: Dict[str, torch.Tensor] = {}
        for vt in self.view_types:
            x = node_features[vt].to(device, non_blocking=True)
            h_items[vt] = self.encoders[vt](x)

        views: List[torch.Tensor] = [h_patient + self.view_type_emb["self"].view(1, -1)]
        count_info: Dict[str, torch.Tensor] = {}
        basic_view_dict: Dict[str, torch.Tensor] = {}
        for vt in self.view_types:
            v, counts = self._aggregate_view(vt, h_patient, h_items[vt], batch_patient_ids, neighbor_maps[vt], device)
            basic_view_dict[vt] = v
            v = v + self.view_type_emb[vt].view(1, -1)
            views.append(v)
            count_info[f"count_{vt}"] = counts

        # No new neighbors or Patient-Item-Patient metapaths are introduced here.
        # These views reorganize already computed Drug/Lab/Procedure representations.
        if self.use_semantic_composite:
            drug_v = basic_view_dict["Drug"]
            lab_v = basic_view_dict["Lab_Item"]
            proc_v = basic_view_dict["Procedure"]
            treatment_v = self.treatment_fusion(torch.cat([drug_v, proc_v], dim=-1))
            evidence_v = self.evidence_fusion(lab_v)
            treatment_v = treatment_v + self.view_type_emb["treatment_event"].view(1, -1)
            evidence_v = evidence_v + self.view_type_emb["evidence"].view(1, -1)
            views.append(treatment_v)
            views.append(evidence_v)

            if self.use_complement_view:
                # complement models the interaction between Lab evidence and treatment events.
                comp_in = torch.cat([evidence_v, treatment_v, torch.abs(evidence_v - treatment_v), evidence_v * treatment_v], dim=-1)
                complement_v = self.complement_fusion(comp_in) + self.view_type_emb["complement"].view(1, -1)
                views.append(complement_v)

        view_stack = torch.stack(views, dim=1)  # [B, num_views, H]
        view_logits = self.view_score(view_stack).squeeze(-1)
        view_alpha = torch.softmax(view_logits, dim=1)
        fused = (view_alpha.unsqueeze(-1) * view_stack).sum(dim=1)
        logits = self.classifier(fused)
        aux = {"view_alpha": view_alpha.detach()}
        aux.update(count_info)
        return logits, aux


# ============================================================
# Training and evaluation
# ============================================================
def iterate_batches(indices: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> List[torch.Tensor]:
    idx = indices.detach().cpu().long()
    if shuffle:
        g = torch.Generator()
        g.manual_seed(int(seed))
        perm = torch.randperm(idx.numel(), generator=g)
        idx = idx[perm]
    return [idx[i:i + batch_size] for i in range(0, idx.numel(), batch_size)]


def evaluate(
    model: nn.Module,
    split: LoadedSplit,
    neighbor_maps: Dict[str, EdgeNeighborMap],
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in iterate_batches(indices, batch_size=batch_size, shuffle=False, seed=0):
            logits, _ = model(split.node_features, batch, neighbor_maps, device)
            logits_list.append(logits.detach().cpu())
            y_list.append(split.labels[batch].detach().cpu())
    logits_all = torch.cat(logits_list, dim=0).numpy()
    y_all = torch.cat(y_list, dim=0).numpy()
    metrics = compute_metrics(y_all, logits_all, split.num_classes)
    return metrics, y_all, logits_all


def train_one_split(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_random_seed(int(args.run_seed))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    split = load_split(args.dataset, args.split, PROJECT_ROOT)
    input_dims = {k: int(v.shape[1]) for k, v in split.node_features.items()}
    for needed in ["Patient", "Drug", "Lab_Item", "Procedure"]:
        if needed not in input_dims:
            raise RuntimeError(f"Missing node type {needed}; ECMF-4DD cannot run.")

    neighbor_maps = {
        "Drug": build_edge_neighbor_map(split.hg, "Drug"),
        "Lab_Item": build_edge_neighbor_map(split.hg, "Lab_Item"),
        "Procedure": build_edge_neighbor_map(split.hg, "Procedure"),
    }
    edge_dims = {vt: nm.edge_attr_dim for vt, nm in neighbor_maps.items()}

    model = ECRDNetV15(
        input_dims=input_dims,
        edge_dims=edge_dims,
        hidden_dim=int(args.hidden_dim),
        num_classes=split.num_classes,
        dropout=float(args.dropout),
        attn_dropout=float(args.attn_dropout),
        use_count_feature=bool(args.use_count_feature),
        use_edge_attr=bool(args.use_edge_attr),
        use_denoise_gate=bool(args.use_denoise_gate),
        idf_bias_scale=float(args.idf_bias_scale),
        use_semantic_composite=bool(args.use_semantic_composite),
        use_complement_view=bool(args.use_complement_view),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ECMF-4DD-model] params={total_params:,}")
    print(f"[ECMF-4DD] training_mode=edge_aware_patient_specific_hop1_full_neighbor_minibatch")
    print(f"[ECMF-4DD] edge_dims={edge_dims}, use_edge_attr={args.use_edge_attr}, use_denoise_gate={args.use_denoise_gate}, use_semantic_composite={args.use_semantic_composite}, use_complement_view={args.use_complement_view}")
    print(f"[ECMF-4DD] batch_size={args.batch_size}, eval_batch_size={args.eval_batch_size}, device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    class_weight = None
    if str(args.loss_mode).lower() == "balanced_ce":
        class_weight = make_class_weight(split.labels, split.train_idx, split.num_classes, power=float(args.class_weight_power)).to(device)
        print(f"[ECMF-4DD-loss] balanced_ce class_weight={class_weight.detach().cpu().tolist()}")

    best_val = -1.0
    best_epoch = -1
    bad_count = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    detail_rows: List[Dict[str, Any]] = []
    start_time = time.time()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []
        for batch in iterate_batches(split.train_idx, int(args.batch_size), shuffle=True, seed=int(args.run_seed) + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(split.node_features, batch, neighbor_maps, device)
            y = split.labels[batch].to(device)
            if class_weight is not None:
                loss = F.cross_entropy(logits, y, weight=class_weight, label_smoothing=float(args.label_smoothing))
            else:
                loss = F.cross_entropy(logits, y, label_smoothing=float(args.label_smoothing))
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        val_metrics, _, _ = evaluate(model, split, neighbor_maps, split.val_idx, int(args.eval_batch_size), device)
        val_score = float(val_metrics[args.early_stop_metric])
        improved = val_score > best_val + float(args.early_stop_min_delta)
        if improved:
            best_val = val_score
            best_epoch = epoch
            bad_count = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_count += 1

        cuda_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_micro_f1": val_metrics["micro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "val_macro_auroc": val_metrics["macro_auroc"],
            "best_epoch": best_epoch,
            "bad_count": bad_count,
            "cuda_peak_mb": cuda_peak,
        }
        detail_rows.append(row)
        if epoch % int(args.log_every) == 0 or epoch == 1:
            print(
                f"[ECMF-4DD][{args.dataset}][{args.split}] "
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                f"val_micro={val_metrics['micro_f1']:.4f} | val_macro={val_metrics['macro_f1']:.4f} | "
                f"val_auprc={val_metrics['macro_auprc']:.4f} | val_auroc={val_metrics['macro_auroc']:.4f} | "
                f"best_{args.early_stop_metric}={best_val:.4f}@{best_epoch} | bad={bad_count} | cuda_peak_mb={cuda_peak:.1f}"
            )
        if bad_count >= int(args.patience):
            print(f"[ECMF-4DD] Early stop at epoch={epoch}, best_epoch={best_epoch}, best_val={best_val:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics, y_test, logits_test = evaluate(model, split, neighbor_maps, split.test_idx, int(args.eval_batch_size), device)
    test_pred = logits_test.argmax(axis=1)
    total_time = time.time() - start_time
    cuda_peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0

    result_dir = Path(args.result_dir)
    ensure_dir(result_dir)
    write_csv_rows(result_dir / "results_detail.csv", detail_rows)
    torch.save(model.state_dict(), result_dir / f"best_model_{args.split}.pt")

    report = classification_report(y_test, test_pred, labels=list(range(split.num_classes)), output_dict=True, zero_division=0)
    report_rows: List[Dict[str, Any]] = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {"label": label}
            row.update(metrics)
            report_rows.append(row)
        else:
            report_rows.append({"label": label, "value": metrics})
    write_csv_rows(result_dir / f"test_classification_report_{args.split}.csv", report_rows)
    write_confusion_matrix(result_dir / f"test_confusion_matrix_{args.split}.csv", y_test, test_pred, split.num_classes)

    summary: Dict[str, Any] = {
        "model": "ECMF-4DD-v1.5",
        "dataset": args.dataset,
        "dataset_name": split.dataset_name,
        "split": args.split,
        "training_mode": "edge_aware_patient_specific_hop1_full_neighbor_minibatch",
        "best_epoch": best_epoch,
        "best_val_metric": args.early_stop_metric,
        "best_val_score": best_val,
        "test_micro_f1": test_metrics["micro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_macro_auprc": test_metrics["macro_auprc"],
        "test_macro_auroc": test_metrics["macro_auroc"],
        "cuda_peak_mb": cuda_peak,
        "time_sec": total_time,
        "params": total_params,
        "hidden_dim": int(args.hidden_dim),
        "edge_dims": edge_dims,
        "use_edge_attr": bool(args.use_edge_attr),
        "use_denoise_gate": bool(args.use_denoise_gate),
        "use_semantic_composite": bool(args.use_semantic_composite),
        "use_complement_view": bool(args.use_complement_view),
        "loss_mode": args.loss_mode,
        "notes": "Ours-v1.5; edge-aware patient-specific hop1 clinical subgraph; evidence/treatment/complement semantic composite views; relation-specific edge encoder; degree/IDF denoising gate; no P-I-P metapath; no global branch.",
    }
    write_json(result_dir / "results_summary.json", summary)
    write_json(result_dir / "run_args.json", vars(args))
    print(
        f"[ECMF-4DD-v1.5][{args.split}] BEST epoch={best_epoch} | "
        f"test_micro={test_metrics['micro_f1']:.4f} | test_macro={test_metrics['macro_f1']:.4f} | "
        f"test_auprc={test_metrics['macro_auprc']:.4f} | test_auroc={test_metrics['macro_auroc']:.4f} | "
        f"cuda_peak_mb={cuda_peak:.1f} | time_sec={total_time:.1f}"
    )
    print(f"[RESULT_DIR] {result_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECMF-4DD v1.5: edge-aware clinical semantic composite relation denoising network")
    parser.add_argument("--dataset", default="mimic3", choices=["mimic3", "mimic4", "data3", "data4", "demo", "sample_demo"])
    parser.add_argument("--split", default="active_split")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-metric", default="macro_f1", choices=["macro_f1", "macro_auprc"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.50)
    parser.add_argument("--attn-dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--use-count-feature", type=int, default=1)
    parser.add_argument("--use-edge-attr", type=int, default=1)
    parser.add_argument("--use-denoise-gate", type=int, default=1)
    parser.add_argument("--idf-bias-scale", type=float, default=0.10)
    parser.add_argument("--use-semantic-composite", type=int, default=1)
    parser.add_argument("--use-complement-view", type=int, default=1)
    parser.add_argument("--loss-mode", default="ce", choices=["ce", "balanced_ce"])
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--run-seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--result-dir", default="")
    args = parser.parse_args()
    if not args.result_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.result_dir = str(PROJECT_ROOT / "results" / "ecmf_4dd" / f"{ts}_{args.dataset}_{args.split}")
    return args


def main() -> None:
    args = parse_args()
    train_one_split(args)


if __name__ == "__main__":
    main()
