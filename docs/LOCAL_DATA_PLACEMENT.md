# Local data placement

This repository does not include real MIMIC-derived graph files. To run the real experiments locally, keep `data3` and `data4` in one of the following layouts.

## Recommended layout when the repository is inside the original project

```text
<project_root>/
├── final_datasets/
│   └── gnn_ready/
│       ├── data3/
│       └── data4/
└── ecmf4dd_github_repo/
    ├── README.md
    ├── data_loading/
    └── experiments/
```

The loader automatically checks the parent directory, so no code change is needed.

## Standalone repository layout

```text
<repo_root>/
├── final_datasets/
│   └── gnn_ready/
│       ├── data3/
│       └── data4/
├── data_loading/
└── experiments/
```

Use this layout if you want the repository to run independently on your machine. Do not upload the real `data3` or `data4` directories to GitHub.

## Demo dataset

A small synthetic demo dataset is included under:

```text
final_datasets/gnn_ready/demo/
```

It is only for smoke testing the code path. It is not used for reported paper results.
