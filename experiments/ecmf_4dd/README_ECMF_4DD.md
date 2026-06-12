# ECMF-4DD training scripts

This directory contains the ECMF-4DD implementation used by the paper. The historical development name in old internal files was ECRD-Net v1.5.

## Main files

- `run_ecmf_4dd.py`: trains and evaluates one dataset split.
- `run_ecmf_4dd_30_splits.py`: launches multiple dataset/split/seed runs and aggregates results.
- `inspect_edge_attrs.py`: checks edge-attribute dimensions and basic statistics.

## Normal local run

Edit the top configuration block in `run_ecmf_4dd_30_splits.py`, then run the file.

Common presets:

- `single_debug`: run one split for a quick check.
- `ratio_7_1_2`: run both real datasets with the 7:1:2 split and five seeds.
- `mimic3_ratio_7_1_2`: run only the MIMIC-III-based dataset.
- `mimic4_ratio_7_1_2`: run only the MIMIC-IV-based dataset.
- `demo`: run the bundled synthetic demo dataset.

The default hyperparameters match the submitted ECMF-4DD experiments.
