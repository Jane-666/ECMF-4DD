# ECMF-4DD

Official code package for **ECMF-4DD: Edge-aware Clinical Multi-view Fusion for Disease Diagnosis**.

The model was developed internally under the name **ECRD-Net v1.5**. The paper and this repository use the final name **ECMF-4DD**. The model computation and hyperparameters are kept unchanged from the submitted experiments.

## Repository layout

```text
ECMF-4DD/
|-- data_loading/                  # unified GNN-ready dataset loader
|-- experiments/ecmf_4dd/          # ECMF-4DD model training scripts
|-- final_datasets/gnn_ready/demo/ # tiny synthetic demo dataset only
|-- results/reported_metrics/      # reported paper metrics
|-- docs/                          # data and reproduction notes
`-- tools/                         # helper scripts
```

## Quick smoke test with the bundled demo dataset

The bundled `demo` dataset is synthetic and contains no real patient data. It only checks whether the code path works.

```bash
python data_loading/gnn_ready_dataset_loader.py --dataset demo --split ratio_7_1_2_seed_1 --backend dgl
python experiments/ecmf_4dd/run_ecmf_4dd.py --dataset demo --split ratio_7_1_2_seed_1 --epochs 2 --patience 1 --device cpu
```

## Running the paper model on your processed data

Place your local GNN-ready datasets under:

```text
final_datasets/gnn_ready/data3/
final_datasets/gnn_ready/data4/
```

Then run:

```bash
python experiments/ecmf_4dd/run_ecmf_4dd_30_splits.py
```

For normal use, edit only the configuration block at the top of `experiments/ecmf_4dd/run_ecmf_4dd_30_splits.py`.

## Data availability and demo dataset

The original MIMIC-III and MIMIC-IV databases are not included in this repository. They are restricted-access clinical datasets hosted on PhysioNet and must be obtained by eligible users after completing the required credentialing process and agreeing to the corresponding data use agreements.

This repository also does not release the processed MIMIC-derived graph tensors, patient-level features, labels, edge attributes, checkpoints, or intermediate files, because these files are derived from restricted-access clinical data.

A small synthetic demo dataset is provided under `final_datasets/gnn_ready/demo` only for code sanity checking. The demo dataset is not derived from MIMIC-III or MIMIC-IV and cannot be used to reproduce the reported experimental results.

Researchers who have authorized access to MIMIC-III and MIMIC-IV may organize their processed data according to the documented directory layout and run the provided scripts to reproduce the experiments.

The data-processing source code used in this study can be provided upon reasonable request, subject to compliance with the PhysioNet data use agreements and institutional requirements.


### Local data placement

The loader uses relative paths by default. For real experiments, it checks these locations in order:

1. `<repo_root>/final_datasets/gnn_ready/data3` and `data4`
2. `<repo_root>/../final_datasets/gnn_ready/data3` and `data4`
3. `<repo_root>/../../final_datasets/gnn_ready/data3` and `data4`

This means the repository can be placed inside an existing project folder, for example:

```text
G:/project_root/
├── final_datasets/gnn_ready/data3
├── final_datasets/gnn_ready/data4
└── ECMF-4DD/
```

In this layout, no data copying is required.
