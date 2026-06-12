# Dataset layout

Real processed datasets should be placed outside public version control or ignored by Git:

```text
final_datasets/gnn_ready/data3/
final_datasets/gnn_ready/data4/
```

Each split directory is expected to contain:

```text
graph.bin              # DGL heterograph, preferred for real experiments
graph_raw_tensors.pt   # optional raw tensor package
graph_schema.json      # schema and summary metadata
```

At the dataset level, the loader expects:

```text
gnn_dataset_registry.csv
```

The bundled `final_datasets/gnn_ready/demo/` directory is a tiny synthetic dataset for smoke tests only. It is not used for paper results.


## Important note

The real processed datasets are not included in this repository. The `data3` and `data4` directories should be kept locally and excluded from public version control. The bundled `demo` directory is synthetic and only verifies the expected file format and execution path.
