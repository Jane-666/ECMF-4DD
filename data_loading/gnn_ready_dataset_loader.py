# -*- coding: utf-8 -*-
"""
Unified loader for ECMF-4DD GNN-ready datasets.

The default layout is:
    <repo_root>/final_datasets/gnn_ready/<dataset>/<split>/

For real experiments, use data3/data4 generated from MIMIC-III/MIMIC-IV.
For repository smoke tests, a small synthetic demo dataset is included.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GNN_READY_ROOT = DEFAULT_PROJECT_ROOT / "final_datasets" / "gnn_ready"

DATASET_NAME_MAP = {
    # Real processed datasets.
    "mimic3": "data3",
    "mimic4": "data4",

    # Direct dataset-directory names.
    "data3": "data3",
    "data4": "data4",

    # Bundled synthetic demo dataset for smoke tests only.
    "demo": "demo",
    "sample_demo": "demo",
}

VALID_RATIOS = {"8_1_1", "7_1_2", "6_2_2"}
VALID_SEEDS = {"1", "10", "100", "1000", "10000"}


@dataclass
class GNNReadyDatasetConfig:
    """Configuration for loading a GNN-ready dataset."""

    project_root: Union[str, Path] = DEFAULT_PROJECT_ROOT
    gnn_ready_root: Optional[Union[str, Path]] = None

    # Use dataset_key such as mimic3/mimic4, or provide dataset_name directly.
    dataset_key: str = "mimic3"
    dataset_name: Optional[str] = None

    # Split selection: split_name, use_active_split, or ratio + seed.
    split_name: Optional[str] = None
    use_active_split: bool = False
    ratio: str = "8_1_1"
    seed: Union[str, int] = 1

    # backend: dgl / raw / both
    backend: str = "dgl"
    device: str = "cpu"

    # Whether to print a loading summary.
    verbose: bool = True


@dataclass
class LoadedGNNReadyDataset:
    """Loaded dataset object."""

    dataset_key: str
    dataset_name: str
    dataset_dir: Path
    split_name: str
    split_dir: Path
    registry: pd.DataFrame
    registry_row: Optional[Dict[str, Any]]
    graph_schema: Dict[str, Any]
    graph: Any = None
    raw_package: Optional[Dict[str, Any]] = None

    def node_types(self) -> List[str]:
        if self.graph is not None and hasattr(self.graph, "ntypes"):
            return list(self.graph.ntypes)
        if self.raw_package is not None:
            return list(self.raw_package.get("node_features", {}).keys())
        return list(self.graph_schema.get("node_summary", {}).keys())

    def canonical_etypes(self) -> List[Any]:
        if self.graph is not None and hasattr(self.graph, "canonical_etypes"):
            return list(self.graph.canonical_etypes)
        if self.raw_package is not None:
            out = []
            for _, e in self.raw_package.get("edges", {}).items():
                out.append(tuple(e.get("canonical_etype", [])))
            return out
        return [tuple(v.get("canonical_etype", [])) for v in self.graph_schema.get("edge_summary", {}).values()]

    def patient_feature_dim(self) -> Optional[int]:
        if self.graph is not None:
            return int(self.graph.nodes["Patient"].data["x"].shape[1])
        if self.raw_package is not None:
            return int(self.raw_package["node_features"]["Patient"]["x"].shape[1])
        info = self.graph_schema.get("node_summary", {}).get("Patient", {})
        return int(info["feature_dim"]) if "feature_dim" in info else None

    def num_classes(self) -> Optional[int]:
        if self.graph is not None:
            y = self.graph.nodes["Patient"].data["y"]
            return int(y.max().item()) + 1 if y.numel() > 0 else 0
        if self.raw_package is not None:
            y = self.raw_package["supervision"]["y"]
            return int(y.max().item()) + 1 if y.numel() > 0 else 0
        sup = self.graph_schema.get("supervision_summary", {})
        return int(sup["num_classes"]) if "num_classes" in sup else None


# -----------------------------
# Path and split resolution
# -----------------------------

def normalize_ratio(ratio: Union[str, Iterable[int]]) -> str:
    """Normalize 8:1:1 / 8-1-1 / 8_1_1 / [8,1,1] into 8_1_1."""
    if isinstance(ratio, (list, tuple)):
        ratio = "_".join(str(x) for x in ratio)
    s = str(ratio).strip().replace(":", "_").replace("-", "_")
    if s.startswith("ratio_"):
        s = s[len("ratio_"):]
    return s


def build_split_name(ratio: Union[str, Iterable[int]] = "8_1_1", seed: Union[str, int] = 1) -> str:
    ratio_s = normalize_ratio(ratio)
    seed_s = str(seed).strip()
    if ratio_s not in VALID_RATIOS:
        raise ValueError(f"Unsupported ratio={ratio_s}. Options: {sorted(VALID_RATIOS)}")
    if seed_s not in VALID_SEEDS:
        raise ValueError(f"Unsupported seed={seed_s}. Options: {sorted(VALID_SEEDS)}")
    return f"ratio_{ratio_s}_seed_{seed_s}"


def resolve_gnn_ready_root(project_root: Union[str, Path], gnn_ready_root: Optional[Union[str, Path]]) -> Path:
    """Resolve the GNN-ready dataset root with safe local fallbacks.

    Preferred layout for a standalone repository:
        <repo_root>/final_datasets/gnn_ready

    If the repository is placed inside an existing project directory, this loader
    also checks the parent project layout:
        <repo_root>/../final_datasets/gnn_ready

    This keeps the code portable while avoiding hard-coded absolute paths.
    """
    if gnn_ready_root is not None:
        root = Path(gnn_ready_root).expanduser()
        return root.resolve() if root.exists() else root

    project_root = Path(project_root).expanduser().resolve()
    candidates = [
        project_root / "final_datasets" / "gnn_ready",
        project_root.parent / "final_datasets" / "gnn_ready",
        project_root.parent.parent / "final_datasets" / "gnn_ready",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Return the preferred path so the final error message is easy to understand.
    return candidates[0]


def resolve_dataset_name(dataset_key: str = "mimic3", dataset_name: Optional[str] = None) -> str:
    if dataset_name:
        return str(dataset_name).strip()
    key = str(dataset_key).strip().lower()
    if key not in DATASET_NAME_MAP:
        raise ValueError(f"Unknown dataset_key={dataset_key}. Options: {sorted(DATASET_NAME_MAP.keys())}. You may also pass dataset_name directly.")
    return DATASET_NAME_MAP[key]


def resolve_dataset_dir(config: GNNReadyDatasetConfig) -> Path:
    root = resolve_gnn_ready_root(config.project_root, config.gnn_ready_root)
    dataset_name = resolve_dataset_name(config.dataset_key, config.dataset_name)
    dataset_dir = root / dataset_name
    if not dataset_dir.exists():
        project_root = Path(config.project_root).expanduser().resolve()
        checked = [
            project_root / "final_datasets" / "gnn_ready" / dataset_name,
            project_root.parent / "final_datasets" / "gnn_ready" / dataset_name,
            project_root.parent.parent / "final_datasets" / "gnn_ready" / dataset_name,
        ]
        checked_msg = "\n".join(f"  - {x}" for x in checked)
        raise FileNotFoundError(
            "Dataset directory does not exist. Checked locations:\n"
            f"{checked_msg}\n\n"
            "For real MIMIC-based experiments, place or link data3/data4 under "
            "final_datasets/gnn_ready. The bundled demo dataset can be loaded with "
            "--dataset demo."
        )
    return dataset_dir


def resolve_split_name(config: GNNReadyDatasetConfig) -> str:
    if config.use_active_split:
        return "active_split"
    if config.split_name:
        s = str(config.split_name).strip()
        if s.lower() in {"active", "active_split"}:
            return "active_split"
        return s
    return build_split_name(config.ratio, config.seed)


def resolve_split_dir(dataset_dir: Union[str, Path], split_name: str) -> Path:
    split_dir = Path(dataset_dir) / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    return split_dir


# -----------------------------
# File loading
# -----------------------------

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(dataset_dir: Union[str, Path]) -> pd.DataFrame:
    path = Path(dataset_dir) / "gnn_dataset_registry.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing gnn_dataset_registry.csv: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def list_formal_splits(dataset_dir: Union[str, Path]) -> List[str]:
    reg = load_registry(dataset_dir)
    if "SPLIT_NAME" not in reg.columns:
        raise ValueError("gnn_dataset_registry.csv is missing the SPLIT_NAME column.")
    return reg["SPLIT_NAME"].astype(str).tolist()


def list_available_split_dirs(dataset_dir: Union[str, Path]) -> List[str]:
    dataset_dir = Path(dataset_dir)
    return sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])


def get_registry_row(registry: pd.DataFrame, split_name: str) -> Optional[Dict[str, Any]]:
    if "SPLIT_NAME" not in registry.columns:
        return None
    target = split_name
    if target == "active_split":
        # active_split is not a formal registry row.
        return None
    hit = registry[registry["SPLIT_NAME"].astype(str) == target]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def safe_torch_load(path: Union[str, Path], device: str = "cpu") -> Dict[str, Any]:
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing torch raw package: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def load_raw_tensor_package(split_dir: Union[str, Path], device: str = "cpu") -> Dict[str, Any]:
    return safe_torch_load(Path(split_dir) / "graph_raw_tensors.pt", device=device)



def build_dgl_graph_from_raw_package(pkg: Dict[str, Any]) -> Any:
    """Build a DGL heterograph from graph_raw_tensors.pt.

    This fallback is mainly used by the bundled synthetic demo dataset. Real
    processed MIMIC-derived datasets usually contain graph.bin directly.
    """
    try:
        import dgl
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError("DGL and PyTorch are required to build a graph from raw tensors.") from exc

    node_features = pkg["node_features"]
    edges = pkg["edges"]
    supervision = pkg.get("supervision", {})

    num_nodes_dict = {ntype: int(info["x"].shape[0]) for ntype, info in node_features.items()}
    graph_data = {}
    edge_attr_by_etype = {}
    for _, info in edges.items():
        etype = tuple(info["canonical_etype"])
        src = torch.as_tensor(info["src"]).long()
        dst = torch.as_tensor(info["dst"]).long()
        graph_data[etype] = (src, dst)
        if "edge_attr" in info:
            edge_attr_by_etype[etype] = torch.as_tensor(info["edge_attr"]).float()

    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    for ntype, info in node_features.items():
        g.nodes[ntype].data["x"] = torch.as_tensor(info["x"]).float()

    if "Patient" in g.ntypes and supervision:
        for key in ["y", "train_mask", "val_mask", "test_mask"]:
            if key in supervision:
                g.nodes["Patient"].data[key] = torch.as_tensor(supervision[key])

    for etype, edge_attr in edge_attr_by_etype.items():
        g.edges[etype].data["edge_attr"] = edge_attr
    return g

def load_dgl_graph(split_dir: Union[str, Path], device: str = "cpu") -> Any:
    try:
        import dgl
    except Exception as exc:
        raise ImportError("DGL cannot be imported. Install DGL or use backend='raw'.") from exc

    split_dir = Path(split_dir)
    graph_path = split_dir / "graph.bin"
    if not graph_path.exists():
        raw_path = split_dir / "graph_raw_tensors.pt"
        if raw_path.exists():
            g = build_dgl_graph_from_raw_package(load_raw_tensor_package(split_dir, device="cpu"))
            if device and str(device).lower() != "cpu":
                g = g.to(device)
            return g
        raise FileNotFoundError(f"Missing graph.bin: {graph_path}")
    graphs, _ = dgl.load_graphs(str(graph_path))
    if not graphs:
        raise RuntimeError(f"dgl.load_graphs returned no graph: {graph_path}")
    g = graphs[0]
    if device and str(device).lower() != "cpu":
        g = g.to(device)
    return g


def load_graph_schema(split_dir: Union[str, Path]) -> Dict[str, Any]:
    return load_json(Path(split_dir) / "graph_schema.json")


# -----------------------------
# Main loading entry point
# -----------------------------

def load_gnn_ready_dataset(config: GNNReadyDatasetConfig) -> LoadedGNNReadyDataset:
    dataset_name = resolve_dataset_name(config.dataset_key, config.dataset_name)
    dataset_dir = resolve_dataset_dir(config)
    split_name = resolve_split_name(config)
    split_dir = resolve_split_dir(dataset_dir, split_name)

    registry = load_registry(dataset_dir)
    registry_row = get_registry_row(registry, split_name)
    schema = load_graph_schema(split_dir)

    backend = str(config.backend).strip().lower()
    if backend not in {"dgl", "raw", "both"}:
        raise ValueError("backend must be one of: dgl / raw / both.")

    graph = None
    raw_package = None

    if backend in {"dgl", "both"}:
        graph = load_dgl_graph(split_dir, device=config.device)
    if backend in {"raw", "both"}:
        raw_package = load_raw_tensor_package(split_dir, device=config.device)

    loaded = LoadedGNNReadyDataset(
        dataset_key=str(config.dataset_key).strip().lower(),
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        split_name=split_name,
        split_dir=split_dir,
        registry=registry,
        registry_row=registry_row,
        graph_schema=schema,
        graph=graph,
        raw_package=raw_package,
    )

    if config.verbose:
        print_loaded_summary(loaded)

    return loaded


def load_from_config_file(path: Union[str, Path]) -> LoadedGNNReadyDataset:
    cfg_dict = load_json(path)
    return load_gnn_ready_dataset(GNNReadyDatasetConfig(**cfg_dict))


# -----------------------------
# Helpers commonly used by training code
# -----------------------------

def extract_dgl_training_tensors(g: Any) -> Tuple[Dict[str, Any], Any, Any, Any, Any]:
    """
    Extract node_features, y, train_mask, val_mask, and test_mask from a DGL graph.
    """
    node_features = {ntype: g.nodes[ntype].data["x"] for ntype in g.ntypes}
    y = g.nodes["Patient"].data["y"]
    train_mask = g.nodes["Patient"].data["train_mask"]
    val_mask = g.nodes["Patient"].data["val_mask"]
    test_mask = g.nodes["Patient"].data["test_mask"]
    return node_features, y, train_mask, val_mask, test_mask


def extract_raw_training_tensors(pkg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Any, Any, Any, Any]:
    """
    Extract training tensors from graph_raw_tensors.pt.
    """
    node_features = {ntype: info["x"] for ntype, info in pkg["node_features"].items()}
    edges = pkg["edges"]
    supervision = pkg["supervision"]
    y = supervision["y"]
    train_mask = supervision["train_mask"]
    val_mask = supervision["val_mask"]
    test_mask = supervision["test_mask"]
    return node_features, edges, y, train_mask, val_mask, test_mask


def get_dgl_input_dims(g: Any) -> Dict[str, int]:
    return {ntype: int(g.nodes[ntype].data["x"].shape[1]) for ntype in g.ntypes}


def get_dgl_edge_attr_dims(g: Any) -> Dict[str, int]:
    out = {}
    for etype in g.canonical_etypes:
        key = "|".join(etype)
        if "edge_attr" in g.edges[etype].data:
            out[key] = int(g.edges[etype].data["edge_attr"].shape[1])
        else:
            out[key] = 0
    return out


def iter_formal_split_configs(
    dataset_key: str = "mimic3",
    dataset_name: Optional[str] = None,
    project_root: Union[str, Path] = DEFAULT_PROJECT_ROOT,
    gnn_ready_root: Optional[Union[str, Path]] = None,
    backend: str = "dgl",
    device: str = "cpu",
    verbose: bool = False,
) -> Iterable[GNNReadyDatasetConfig]:
    """Iterate over formal split configurations listed in the registry."""
    tmp_cfg = GNNReadyDatasetConfig(
        project_root=project_root,
        gnn_ready_root=gnn_ready_root,
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        backend=backend,
        device=device,
        verbose=verbose,
    )
    dataset_dir = resolve_dataset_dir(tmp_cfg)
    for split_name in list_formal_splits(dataset_dir):
        yield GNNReadyDatasetConfig(
            project_root=project_root,
            gnn_ready_root=gnn_ready_root,
            dataset_key=dataset_key,
            dataset_name=dataset_name,
            split_name=split_name,
            backend=backend,
            device=device,
            verbose=verbose,
        )


def print_loaded_summary(loaded: LoadedGNNReadyDataset) -> None:
    print("=" * 80)
    print("GNN-ready dataset loaded")
    print("=" * 80)
    print("dataset_key  :", loaded.dataset_key)
    print("dataset_name :", loaded.dataset_name)
    print("dataset_dir  :", loaded.dataset_dir)
    print("split_name   :", loaded.split_name)
    print("split_dir    :", loaded.split_dir)
    print("node_types   :", loaded.node_types())
    print("canonical_etypes:")
    for et in loaded.canonical_etypes():
        print("  -", et)
    print("patient_dim  :", loaded.patient_feature_dim())
    print("num_classes  :", loaded.num_classes())

    if loaded.graph is not None:
        print("backend      : DGL")
        print("num_nodes    :", {ntype: int(loaded.graph.num_nodes(ntype)) for ntype in loaded.graph.ntypes})
        print("num_edges    :", {"|".join(et): int(loaded.graph.num_edges(et)) for et in loaded.graph.canonical_etypes})
        print("input_dims   :", get_dgl_input_dims(loaded.graph))
        print("edge_dims    :", get_dgl_edge_attr_dims(loaded.graph))
    if loaded.raw_package is not None:
        print("backend      : RAW")
        print("raw keys     :", list(loaded.raw_package.keys()))
    print("=" * 80)


# -----------------------------
# Command-line interface
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load or inspect an ECMF-4DD GNN-ready dataset.")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT), help="Project root. Default: repository root inferred from this file.")
    parser.add_argument("--gnn-ready-root", default="", help="GNN-ready root. Default: <project-root>/final_datasets/gnn_ready")
    parser.add_argument("--dataset", default="mimic3", help="mimic3 / mimic4 / data3 / data4 / demo")
    parser.add_argument("--dataset-name", default="", help="Full dataset directory name. Overrides --dataset when provided.")
    parser.add_argument("--split", default="active", help="active / active_split / ratio_8_1_1_seed_1; ratio+seed is used when omitted")
    parser.add_argument("--ratio", default="8_1_1", help="8_1_1 / 7_1_2 / 6_2_2")
    parser.add_argument("--seed", default="1", help="1 / 10 / 100 / 1000 / 10000")
    parser.add_argument("--backend", default="dgl", choices=["dgl", "raw", "both"], help="Load graph.bin or raw tensor package")
    parser.add_argument("--device", default="cpu", help="cpu / cuda")
    parser.add_argument("--list-splits", action="store_true", help="Only list formal and available splits without loading a graph")
    parser.add_argument("--config", default="", help="Load from a JSON config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        load_from_config_file(args.config)
        return

    split_arg = str(args.split).strip()
    use_active = split_arg.lower() in {"active", "active_split"}
    split_name = None if use_active or not split_arg else split_arg

    config = GNNReadyDatasetConfig(
        project_root=args.project_root,
        gnn_ready_root=args.gnn_ready_root or None,
        dataset_key=args.dataset,
        dataset_name=args.dataset_name or None,
        split_name=split_name,
        use_active_split=use_active,
        ratio=args.ratio,
        seed=args.seed,
        backend=args.backend,
        device=args.device,
        verbose=True,
    )

    dataset_dir = resolve_dataset_dir(config)

    if args.list_splits:
        print("dataset_dir:", dataset_dir)
        print("formal splits from registry:")
        for s in list_formal_splits(dataset_dir):
            print("  -", s)
        print("available split directories:")
        for s in list_available_split_dirs(dataset_dir):
            print("  -", s)
        return

    load_gnn_ready_dataset(config)


if __name__ == "__main__":
    main()
