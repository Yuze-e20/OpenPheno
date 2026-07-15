# Cell Count Baselines

This directory contains the released Cell count baseline scripts and result files used as simple morphology-derived controls.

The baseline feature is `Cells_Number_Object_Number`, stored in:

```text
src/cellcount/resources/CP_count_PUMA.csv
```

## Scripts

```text
run_cellcount270_lr.py
```

Runs a per-assay logistic regression Cell count baseline on the Broad-270 five-fold compound splits.

```text
run_cellcount_setting2_fewshot_lr.py
```

Runs the per-assay logistic regression Cell count baseline for the Setting 2 few-shot splits.

```text
run_setting2_nearest_cellcount_threshold.py
```

Runs the Setting 2 zero-shot Cell count threshold-transfer control. For each held-out assay, the threshold is selected on the nearest training assay and then applied to the held-out assay.

```text
reproduce_seal_sanchez_logreg.py
```

Reproduces the logistic-regression Cell count baseline on the Sanchez-Fernandez/CLOOME-style files when those external files are available under `src/cellcount/resources/The_Sanchez-Fernandez_Files`.

## Released Results

```text
results/cellcount270_per_assay_lr/
results/setting2_fewshot_curve_trainonly_cellcount_lr/
results/setting2_nearest_cellcount_threshold/
results/reproduce_seal_sanchez_logreg/
```

The released CSV files include per-assay metrics, logits where available, and summary tables.
