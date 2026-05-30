# CrabNet MatBench Evaluation and WBM Residual Calibration

This document summarizes the later-stage experiments in this repository. The
work is organized in two directories:

```text
crabnet/
residual model/
```

- `crabnet/` contains the MatBench composition-property experiments using
  CrabNet with Mat2Vec, MDS-derived embeddings, and Mat2Vec+MDS concatenated
  embeddings.
- `residual model/` contains the WBM out-of-distribution experiments, including
  frozen baseline prediction and embedding-based residual calibration.

---

## 1. Data Sources

The datasets used in these experiments were obtained from two publicly
available sources:

- MatBench dataset: https://matbench.materialsproject.org/
- WBM benchmark test set in Matbench Discovery:
  https://matbench-discovery.materialsproject.org/data

The local WBM files used by the residual-calibration experiments are stored in:

```text
residual model/data/
```

This directory contains:

```text
2022-10-19-wbm-computed-structure-entries.json
wbm_summary.csv.gz
```

The WBM summary file contains 256,963 materials and is used together with the
computed-structure-entry JSON file for full WBM baseline prediction and
residual-calibration analysis.

---

## 2. Environment Setup

The supplemental experiments should be run in separate Python environments.
Using separate environments is recommended because CrabNet/MatBench,
Matbench Discovery, and the frozen baseline models have different dependency
requirements.

### 2.1 CrabNet / MatBench Environment

Create this environment for the experiments in `crabnet/`:

```bash
conda create -n crabnet2 python=3.8
conda activate crabnet2
pip install -r requirements-crabnet.txt
```

This environment is used for:

```bash
cd crabnet
bash train_matbench_emb.sh
bash train_matbench_mat2vec_mds_concat.sh
```

### 2.2 WBM Residual-calibration Environment

Create this environment for residual-calibration analysis after the frozen
baseline prediction files have been generated:

```bash
conda create -n matbench-discovery python=3.11
conda activate matbench-discovery
pip install -r requirements-wbm-residual.txt
```

This environment is used for:

```bash
cd "residual model/frozen baseline"
bash convert_all_energy_to_eform.sh

cd "../mat2vec_residual"
bash run_all_embedding_residuals.sh
```

### 2.3 Frozen-baseline Prediction Environments

Frozen-baseline WBM prediction should be run in model-specific environments.
Create and activate the corresponding environment, install the matching
requirements file, and then run the selected baseline model.

| Baseline model | Suggested environment | Python | Requirements file |
| --- | --- | --- | --- |
| `chgnet` | `chgnet-wbm` | 3.11 | `requirements-baseline-chgnet.txt` |
| `mace-mpa-0` | `mace-wbm` | 3.11 | `requirements-baseline-mace.txt` |
| `m3gnet` | `m3gnet-wbm` | 3.9 | `requirements-baseline-m3gnet.txt` |
| `mattersim-v1-5m` | `mattersim-wbm` | 3.9 | `requirements-baseline-mattersim.txt` |
| `orb-v3` | `orb-wbm` | 3.11 | `requirements-baseline-orb.txt` |
| `sevennet-l3i5` | `sevennet-wbm` | 3.11 | `requirements-baseline-sevennet.txt` |

Example for CHGNet:

```bash
conda create -n chgnet-wbm python=3.11
conda activate chgnet-wbm
pip install -r requirements-baseline-chgnet.txt

cd "residual model/frozen baseline"
bash run_all_baseline_wbm_full.sh chgnet
```

Use the same pattern for the other baseline models by changing the environment
name, requirements file, and model key. Running
`bash run_all_baseline_wbm_full.sh` without a model argument requires all
baseline model packages to be installed in the same environment, which is not
recommended.

---

## 3. CrabNet Experiments on MatBench

The CrabNet experiments are located in:

```text
crabnet/
```

These experiments evaluate different element embedding tables on selected
MatBench composition tasks.

### 3.1 Embedding Tables

All embedding tables used by CrabNet are stored in:

```text
crabnet/data/element_properties/
```

The main embedding files include:

```text
mat2vec.csv
MDS_32_cos_zscore.csv
MDS_64_cos_zscore.csv
MDS_E_32_cos_zscore.csv
MDS_E_64_cos_zscore.csv
MDS_F_32_cos_zscore.csv
MDS_F_64_cos_zscore.csv
mat2vec+S32.csv
mat2vec+S64.csv
mat2vec+L32.csv
mat2vec+L64.csv
mat2vec+H32.csv
mat2vec+H64.csv
```

The Mat2Vec+MDS concatenated embeddings can be regenerated with:

```bash
cd crabnet
python build_mat2vec_mds_concat_embedding.py
```

This script reads `mat2vec.csv` and the corresponding MDS embedding files,
keeps their common element set, concatenates the feature dimensions, and writes
the resulting Mat2Vec+S/L/H embedding tables back to
`crabnet/data/element_properties/`.

### 3.2 Batch CrabNet Runs

To evaluate the original Mat2Vec and MDS embeddings:

```bash
cd crabnet
bash train_matbench_emb.sh
```

To evaluate the Mat2Vec+MDS concatenated embeddings:

```bash
cd crabnet
bash train_matbench_mat2vec_mds_concat.sh
```

---

## 4. Frozen Baseline Prediction on WBM

The frozen-baseline WBM experiments are located in:

```text
residual model/frozen baseline/
```

### 4.1 Supported Baseline Models

The supported baseline model keys are:

```text
chgnet
m3gnet
mace-mpa-0
mattersim-v1-5m
orb-v3
sevennet-l3i5
```

### 4.2 Raw WBM Prediction

Example for CHGNet:

```bash
conda activate chgnet-wbm
cd "residual model/frozen baseline"
bash run_all_baseline_wbm_full.sh chgnet
```

Example for MACE:

```bash
conda activate mace-wbm
cd "residual model/frozen baseline"
bash run_all_baseline_wbm_full.sh mace-mpa-0
```

### 4.3 Formation-energy Conversion

After raw energy prediction, convert all available model outputs to
formation-energy CSV files:

```bash
conda activate matbench-discovery
cd "residual model/frozen baseline"
bash convert_all_energy_to_eform.sh
```

---

## 5. WBM Residual Calibration with Element Embeddings

The residual-calibration experiments are located in:

```text
residual model/mat2vec_residual/
```

The goal is to correct frozen baseline formation-energy predictions on WBM
using a lightweight residual model based on element embeddings.

For each material, the residual target is:

```text
residual = true formation energy - frozen baseline predicted formation energy
```

### 5.1 Inputs

The residual-calibration workflow uses:

```text
residual model/data/wbm_summary.csv.gz
residual model/frozen baseline/outputs/<baseline-model>/eform/<baseline-model>_wbm_computed_full.csv
residual model/embeddings/
```

The embedding directory contains:

```text
mat2vec.csv
MDS_32_cos_zscore.csv
MDS_64_cos_zscore.csv
MDS_E_32_cos_zscore.csv
MDS_E_64_cos_zscore.csv
MDS_F_32_cos_zscore.csv
MDS_F_64_cos_zscore.csv
mat2vec+S32.csv
mat2vec+S64.csv
mat2vec+L32.csv
mat2vec+L64.csv
mat2vec+H32.csv
mat2vec+H64.csv
```

### 5.2 Run All Residual-calibration Experiments

To run all supported baselines and all embedding tables:

```bash
conda activate matbench-discovery
cd "residual model/mat2vec_residual"
bash run_all_embedding_residuals.sh
```
