# Running ECMF-4DD with Authorized Data

This repository does not include the original MIMIC-III or MIMIC-IV databases, nor does it include processed MIMIC-derived graph tensors, patient-level features, labels, edge attributes, checkpoints, or intermediate files.

The bundled demo dataset is a small synthetic dataset for code sanity checking only. It is not derived from MIMIC-III or MIMIC-IV and cannot reproduce the reported experimental results.

## Data access

Researchers who want to reproduce the reported experiments should first obtain authorized access to MIMIC-III and MIMIC-IV through PhysioNet and comply with the corresponding data use agreements and institutional requirements.

The data-processing source code used in this study can be provided upon reasonable request, subject to compliance with the PhysioNet data use agreements and institutional requirements.

## Expected processed-data layout

After processing the authorized data, place the processed graph datasets under:

```text
final_datasets/gnn_ready/
```

A typical local layout is:

```text
final_datasets/
└── gnn_ready/
    ├── data3/
    │   ├── ratio_7_1_2_seed_1/
    │   ├── ratio_7_1_2_seed_10/
    │   ├── ratio_7_1_2_seed_100/
    │   ├── ratio_7_1_2_seed_1000/
    │   └── ratio_7_1_2_seed_10000/
    └── data4/
        ├── ratio_7_1_2_seed_1/
        ├── ratio_7_1_2_seed_10/
        ├── ratio_7_1_2_seed_100/
        ├── ratio_7_1_2_seed_1000/
        └── ratio_7_1_2_seed_10000/
```

Here, `data3` and `data4` are only local engineering aliases for the processed MIMIC-III-based and MIMIC-IV-based datasets. They are not official dataset names.

Each split directory is expected to contain the processed graph files required by the loader, such as graph schema files, graph tensors, node features, labels, split masks, and edge attributes, depending on the exported data format.

## Running the main script

For a single local test run, use the runner configuration in:

```text
experiments/ecmf_4dd/run_ecmf_4dd_30_splits.py
```

The default configuration is intended to run one split first:

```text
RUN_PRESET     = "single_debug"
SINGLE_DATASET = "mimic4"
SINGLE_RATIO   = "7_1_2"
SINGLE_SEED    = "1"
```

For full reported experiments, use the `ratio_7_1_2` preset after all required split directories are available.
