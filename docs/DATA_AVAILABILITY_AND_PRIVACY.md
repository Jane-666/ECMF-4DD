# Data availability and privacy

The original MIMIC-III and MIMIC-IV databases are not included in this repository. They are restricted-access clinical datasets hosted on PhysioNet and must be obtained by eligible users after completing the required credentialing process and agreeing to the corresponding data use agreements.

This repository also does not release the processed MIMIC-derived graph tensors, patient-level features, labels, edge attributes, checkpoints, or intermediate files, because these files are derived from restricted-access clinical data.

A small synthetic demo dataset is provided under `final_datasets/gnn_ready/demo` only for code sanity checking. The demo dataset is not derived from MIMIC-III or MIMIC-IV and cannot be used to reproduce the reported experimental results.

Researchers who have authorized access to MIMIC-III and MIMIC-IV may organize their processed data according to the documented directory layout and run the provided scripts to reproduce the experiments.

The data-processing source code used in this study can be provided upon reasonable request, subject to compliance with the PhysioNet data use agreements and institutional requirements.
