#!/usr/bin/env python
"""Per-assay logistic-regression cell-count baseline on Broad-270 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELLCOUNT_CSV = PROJECT_ROOT / "src/cellcount/resources/CP_count_PUMA.csv"
DEFAULT_LABEL_CSV = PROJECT_ROOT / "src/data/Broad-270/label/compound_assay_matrix_270.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "src/data/Broad-270/compound_splits"
DEFAULT_OUT_DIR = PROJECT_ROOT / "src/cellcount/results/cellcount270_per_assay_lr"


def parse_folds(value: str) -> list[int]:
    folds = [int(x) for x in value.split(",") if x.strip()]
    if not folds:
        raise argparse.ArgumentTypeError("fold list is empty")
    return folds


def load_split_smiles(split_dir: Path, prefix: str, fold: int) -> set[str]:
    split = pd.read_csv(split_dir / f"{prefix}{fold}.csv")
    if "Compound SMILES" not in split.columns:
        raise ValueError(f"{prefix}{fold}.csv missing 'Compound SMILES'")
    return set(split["Compound SMILES"].dropna().astype(str))


def fit_lr(x: np.ndarray, y: np.ndarray) -> Pipeline | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, class_weight="balanced")),
        ]
    )
    return model.fit(x.reshape(-1, 1), y)


def positive_probability(model: Pipeline, x: np.ndarray) -> np.ndarray:
    clf = model.named_steps["classifier"]
    proba = model.predict_proba(x.reshape(-1, 1))
    class_to_index = {int(cls): idx for idx, cls in enumerate(clf.classes_)}
    if 1 not in class_to_index:
        return np.zeros(len(x), dtype=float)
    return proba[:, class_to_index[1]]


def summarize(metrics: pd.DataFrame, assays: list[str], out_dir: Path) -> dict:
    valid = metrics.dropna(subset=["auroc"])
    observed = (
        valid.groupby("assay_id", as_index=False)["auroc"]
        .median()
        .rename(columns={"auroc": "median_auroc"})
    )
    all_assays = pd.DataFrame({"assay_id": assays})
    assay_summary = all_assays.merge(observed, on="assay_id", how="left")
    assay_summary["median_auroc_filled"] = assay_summary["median_auroc"].fillna(0.5)
    assay_summary.to_csv(out_dir / "assay_median_auroc_lr.csv", index=False)

    fold_counts = (
        metrics.assign(auroc_filled=metrics["auroc"].fillna(0.5))
        .groupby("fold")
        .agg(
            n_valid_auroc=("auroc", lambda x: int(x.notna().sum())),
            count_gt_0_9=("auroc_filled", lambda x: int((x > 0.9).sum())),
            count_gt_0_7=("auroc_filled", lambda x: int((x > 0.7).sum())),
            count_gt_0_5=("auroc_filled", lambda x: int((x > 0.5).sum())),
        )
        .reset_index()
    )
    fold_counts.to_csv(out_dir / "threshold_counts_by_fold_lr.csv", index=False)

    filled = assay_summary["median_auroc_filled"]
    observed_values = assay_summary["median_auroc"].dropna()
    summary = {
        "model": "per_assay_logistic_regression",
        "feature": "Cells_Number_Object_Number",
        "n_assays": int(len(assays)),
        "n_assays_with_valid_fold": int(assay_summary["median_auroc"].notna().sum()),
        "n_valid_fold_aurocs": int(metrics["auroc"].notna().sum()),
        "mean_observed_assay_median_auroc": float(observed_values.mean()) if len(observed_values) else np.nan,
        "mean_270_assay_median_auroc_fill_missing_0.5": float(filled.mean()),
        "median_count_gt_0.9": int((filled > 0.9).sum()),
        "median_count_gt_0.7": int((filled > 0.7).sum()),
        "median_count_gt_0.5": int((filled > 0.5).sum()),
        "avg_fold_count_gt_0.9": float(fold_counts["count_gt_0_9"].mean()),
        "avg_fold_count_gt_0.7": float(fold_counts["count_gt_0_7"].mean()),
        "avg_fold_count_gt_0.5": float(fold_counts["count_gt_0_5"].mean()),
        "fallback_auroc": 0.5,
    }
    (out_dir / "summary_lr.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(out_dir / "summary_lr.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellcount-csv", type=Path, default=DEFAULT_CELLCOUNT_CSV)
    parser.add_argument("--label-csv", type=Path, default=DEFAULT_LABEL_CSV)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("1,2,3,4,5"))
    parser.add_argument("--feature", default="Cells_Number_Object_Number")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.label_csv)
    cellcount = pd.read_csv(args.cellcount_csv)
    assays = labels.columns[1:].astype(str).tolist()
    data = labels.merge(cellcount[["smiles", args.feature]], on="smiles", how="left")
    if data[args.feature].isna().any():
        raise ValueError("Some compounds have no cell-count feature")

    rows = []
    logits = []
    for fold in args.folds:
        train_smiles = load_split_smiles(args.split_dir, "train", fold)
        test_smiles = load_split_smiles(args.split_dir, "val", fold)
        train = data[data["smiles"].isin(train_smiles)]
        test = data[data["smiles"].isin(test_smiles)]

        for assay in assays:
            train_assay = train[["smiles", args.feature, assay]].rename(columns={assay: "label"})
            train_assay = train_assay.dropna(subset=[args.feature, "label"])
            test_assay = test[["smiles", args.feature, assay]].rename(columns={assay: "label"})
            test_assay = test_assay.dropna(subset=[args.feature, "label"])

            y_train = train_assay["label"].astype(int).to_numpy()
            x_train = train_assay[args.feature].astype(float).to_numpy()
            y_test = test_assay["label"].astype(int).to_numpy()
            x_test = test_assay[args.feature].astype(float).to_numpy()
            model = fit_lr(x_train, y_train)

            row = {
                "fold": fold,
                "assay_id": assay,
                "auroc": np.nan,
                "auprc": np.nan,
                "test_set_length": int(len(y_test)),
                "num_actives": int((y_test == 1).sum()),
                "num_inactives": int((y_test == 0).sum()),
                "n_train_labels": int(len(y_train)),
                "n_train_pos": int((y_train == 1).sum()),
                "n_train_neg": int((y_train == 0).sum()),
                "feature": args.feature,
                "model": "per_assay_lr",
                "constant_fallback": True,
            }
            if model is not None and len(y_test) > 0 and len(np.unique(y_test)) >= 2:
                score = positive_probability(model, x_test)
                row["auroc"] = float(roc_auc_score(y_test, score))
                row["auprc"] = float(average_precision_score(y_test, score))
                row["constant_fallback"] = False
                logit = test_assay[["smiles", "label"]].copy()
                logit["fold"] = fold
                logit["assay_id"] = assay
                logit["score"] = score
                logits.append(logit[["fold", "assay_id", "smiles", "score", "label"]])
            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "all_folds_val_metrics_lr.csv", index=False)
    if logits:
        pd.concat(logits, ignore_index=True).to_csv(args.out_dir / "logits_lr.csv", index=False)
    print(json.dumps(summarize(metrics, assays, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
