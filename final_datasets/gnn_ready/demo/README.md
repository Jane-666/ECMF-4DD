# Synthetic demo dataset

This directory contains a tiny synthetic graph for smoke tests. It contains no real patient data and is not used for paper results.

Run:

```bash
python data_loading/gnn_ready_dataset_loader.py --dataset demo --split ratio_7_1_2_seed_1 --backend dgl
python experiments/ecmf_4dd/run_ecmf_4dd.py --dataset demo --split ratio_7_1_2_seed_1 --epochs 2 --patience 1 --device cpu
```
