# Reproducibility

## Smoke test

```bash
python data_loading/gnn_ready_dataset_loader.py --dataset demo --split ratio_7_1_2_seed_1 --backend dgl
python experiments/ecmf_4dd/run_ecmf_4dd.py --dataset demo --split ratio_7_1_2_seed_1 --epochs 2 --patience 1 --device cpu
```

## Paper setting

Place the processed GNN-ready datasets under:

```text
final_datasets/gnn_ready/data3/
final_datasets/gnn_ready/data4/
```

Then edit `RUN_PRESET` in:

```text
experiments/ecmf_4dd/run_ecmf_4dd_30_splits.py
```

Use `ratio_7_1_2` to run the main five-seed setting on both datasets.


## Data-processing scripts

The data-processing source code used in this study can be provided upon reasonable request, subject to compliance with the PhysioNet data use agreements and institutional requirements.
