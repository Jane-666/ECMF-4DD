# Experiment Configuration

This document summarizes the main experimental configuration used for ECMF-4DD.

## Task setting

ECMF-4DD is used for Patient node disease classification on medical heterogeneous graphs constructed from structured electronic health records. The graph contains four node types: Patient, Drug, Lab_Item, and Procedure. Drug, Lab_Item, and Procedure are treated as clinical items directly associated with Patient nodes.

The reported experiments use two processed graph datasets derived from MIMIC-III and MIMIC-IV. In the local implementation, the directory names `data3` and `data4` are only engineering aliases for the processed MIMIC-III-based and MIMIC-IV-based datasets.

## Main split and seeds

The main experiments use the following data split setting:

```text
ratio_7_1_2
```

This corresponds to training, validation, and test sets with an approximate ratio of 7:1:2.

The reported multi-run results are based on five random seeds:

```text
1, 10, 100, 1000, 10000
```

## Main hyperparameters

The default hyperparameters used for the reported ECMF-4DD results are:

```text
hidden_dim           = 128
dropout              = 0.50
attn_dropout         = 0.10
learning_rate        = 0.001
weight_decay         = 0.0001
batch_size           = 512
eval_batch_size      = 2048
max_epochs           = 300
patience             = 30
early_stop_metric    = macro_f1
early_stop_min_delta = 0.0001
grad_clip            = 1.0
loss_mode            = ce
class_weight_power   = 0.5
label_smoothing      = 0.0
```

## Enabled model components

The reported ECMF-4DD configuration enables the following model components:

```text
use_count_feature       = 1
use_edge_attr           = 1
use_denoise_gate        = 1
use_semantic_composite  = 1
use_complement_view     = 1
idf_bias_scale          = 0.1
```

These settings correspond to the complete ECMF-4DD model described in the manuscript.

## Evaluation metrics

The reported metrics are:

```text
Micro-F1
Macro-F1
Macro-AUPRC
Macro-AUROC
```

Validation Macro-F1 is used for early stopping and model selection.
