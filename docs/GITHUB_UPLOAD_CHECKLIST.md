# GitHub upload checklist

Before making the repository public, check that the following files or directories are not tracked by Git:

- real `final_datasets/gnn_ready/data3/`
- real `final_datasets/gnn_ready/data4/`
- raw MIMIC-III or MIMIC-IV tables
- patient-level tensors, labels, edge attributes, CSV files, or DGL graph files derived from MIMIC
- training checkpoints such as `*.pth`, `*.pt`, or `*.ckpt` generated from real MIMIC-derived data
- runtime outputs under `results/ecmf_4dd/`
- local absolute-path logs

It is safe to upload the synthetic `final_datasets/gnn_ready/demo/` directory, because it is not derived from MIMIC-III or MIMIC-IV and is only used for code sanity checking.
