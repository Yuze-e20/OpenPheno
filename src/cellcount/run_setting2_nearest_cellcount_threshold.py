#!/usr/bin/env python
"""Setting 2 cell-count threshold transfer control.

For each held-out Broad-270 assay, identify the nearest training assay using the
BioLord nearest-assay audit, choose the best one-feature cell-count threshold on
that training assay, and transfer the same threshold/direction to the held-out
assay. This matches the unseen-assay control setting: no labels from the target
assay are used to tune the cell-count rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cellcount-csv",
        type=Path,
        default=PROJECT_ROOT / "src/cellcount/resources/CP_count_PUMA.csv",
    )
    parser.add_argument(
        "--label-csv",
        type=Path,
        default=PROJECT_ROOT / "src/data/Broad-270/label/compound_assay_matrix_270.csv",
    )
    parser.add_argument(
        "--test-split",
        type=Path,
        default=PROJECT_ROOT / "src/data/Broad-270/assay_splits/fewshotold/label_test.csv",
    )
    parser.add_argument(
        "--assay-meta",
        type=Path,
        default=PROJECT_ROOT / "src/data/Broad-270/assay_meta/assay_meta.csv",
    )
    parser.add_argument(
        "--nearest-summary",
        type=Path,
        default=PROJECT_ROOT / "src/cellcount/resources/setting2_assay_overlap_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "src/cellcount/results/setting2_nearest_cellcount_threshold",
    )
    parser.add_argument("--feature", default="Cells_Number_Object_Number")
    return parser.parse_args()


def read_assay_ids(path: Path) -> list[str]:
    return [c for c in pd.read_csv(path, nrows=0).columns if c != "smiles"]


def candidate_thresholds(x: np.ndarray) -> np.ndarray:
    vals = np.unique(np.asarray(x, dtype=float))
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.array([], dtype=float)
    if len(vals) == 1:
        return vals
    mids = (vals[:-1] + vals[1:]) / 2.0
    return np.concatenate(([vals[0] - 1e-9], mids, [vals[-1] + 1e-9]))


def choose_threshold(y: np.ndarray, x: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "threshold": np.nan,
            "direction": "",
            "train_balanced_accuracy": np.nan,
            "train_binary_auroc": np.nan,
        }

    best = {
        "threshold": np.nan,
        "direction": "",
        "train_balanced_accuracy": -np.inf,
        "train_binary_auroc": np.nan,
    }
    for threshold in candidate_thresholds(x):
        for direction in ("high_cell_count_active", "low_cell_count_active"):
            pred = apply_threshold(x, threshold, direction)
            bal = balanced_accuracy_score(y, pred)
            if bal > best["train_balanced_accuracy"]:
                best = {
                    "threshold": float(threshold),
                    "direction": direction,
                    "train_balanced_accuracy": float(bal),
                    "train_binary_auroc": float(roc_auc_score(y, pred)),
                }
    return best


def apply_threshold(x: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == "high_cell_count_active":
        return (x > threshold).astype(int)
    if direction == "low_cell_count_active":
        return (x <= threshold).astype(int)
    raise ValueError(f"unknown direction: {direction}")


def eval_binary_predictions(y: np.ndarray, pred: np.ndarray) -> dict:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "balanced_accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
        }
    return {
        "auroc": float(roc_auc_score(y, pred)),
        "auprc": float(average_precision_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def summarize(metrics: pd.DataFrame) -> dict:
    filled = metrics["auroc"].fillna(0.5)
    valid = metrics.dropna(subset=["auroc"])
    return {
        "n_assays": int(len(metrics)),
        "n_valid_assays": int(len(valid)),
        "n_fallback_assays": int(metrics["auroc"].isna().sum()),
        "mean_auroc_valid": float(valid["auroc"].mean()) if len(valid) else float("nan"),
        "median_auroc_valid": float(valid["auroc"].median()) if len(valid) else float("nan"),
        "mean_auroc": float(filled.mean()),
        "median_auroc": float(filled.median()),
        "auroc_gt_0.9": int((filled > 0.9).sum()),
        "auroc_gt_0.7": int((filled > 0.7).sum()),
        "auroc_gt_0.5": int((filled > 0.5).sum()),
        "fallback_auroc": 0.5,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_ids = read_assay_ids(args.test_split)
    labels = pd.read_csv(args.label_csv)
    cellcount = pd.read_csv(args.cellcount_csv)
    meta = pd.read_csv(args.assay_meta, dtype={"PUMA_ASSAY_ID": str})
    nearest = pd.read_csv(args.nearest_summary, dtype={"assay_id": str})

    if args.feature not in cellcount.columns:
        raise ValueError(f"feature column not found: {args.feature}")
    if labels["smiles"].duplicated().any():
        raise ValueError("label csv contains duplicated smiles")
    if cellcount["smiles"].duplicated().any():
        raise ValueError("cellcount csv contains duplicated smiles")

    nearest_map = nearest.set_index("assay_id")["nearest_biolord_train_assay_id"].astype(str).to_dict()
    nearest_sim = nearest.set_index("assay_id")["nearest_biolord_cosine"].to_dict()
    missing_nearest = sorted(set(test_ids) - set(nearest_map))
    if missing_nearest:
        raise ValueError(f"nearest summary is missing held-out assays: {missing_nearest}")

    needed_train_ids = sorted({nearest_map[assay] for assay in test_ids})
    data = labels[["smiles", *test_ids, *needed_train_ids]].merge(
        cellcount[["smiles", args.feature]], on="smiles", how="left"
    )
    missing_feature = int(data[args.feature].isna().sum())
    if missing_feature:
        raise ValueError(f"{missing_feature} compounds have no cell-count feature")

    assay_type = meta.set_index("PUMA_ASSAY_ID")["ASSAY_TYPE"].to_dict()
    assay_name = meta.set_index("PUMA_ASSAY_ID")["ASSAY_NAME"].to_dict()

    rows = []
    logits = []
    for test_id in test_ids:
        train_id = nearest_map[test_id]
        train_df = data[["smiles", args.feature, train_id]].rename(columns={train_id: "label"})
        train_df = train_df.dropna(subset=[args.feature, "label"])
        y_train = train_df["label"].astype(int).to_numpy()
        x_train = train_df[args.feature].astype(float).to_numpy()
        rule = choose_threshold(y_train, x_train)

        test_df = data[["smiles", args.feature, test_id]].rename(columns={test_id: "label"})
        test_df = test_df.dropna(subset=[args.feature, "label"])
        y_test = test_df["label"].astype(int).to_numpy()
        x_test = test_df[args.feature].astype(float).to_numpy()

        if rule["direction"]:
            pred = apply_threshold(x_test, rule["threshold"], rule["direction"])
            metrics = eval_binary_predictions(y_test, pred)
            logit = test_df[["smiles", "label"]].copy()
            logit["assay_id"] = test_id
            logit["score"] = pred
            logit["nearest_train_assay_id"] = train_id
            logit["threshold"] = rule["threshold"]
            logit["direction"] = rule["direction"]
            logits.append(logit)
        else:
            metrics = eval_binary_predictions(y_test, np.zeros(len(y_test), dtype=int))

        rows.append(
            {
                "assay_id": test_id,
                "assay_type": assay_type.get(test_id, "unknown"),
                "assay_name": assay_name.get(test_id, ""),
                "nearest_train_assay_id": train_id,
                "nearest_train_assay_type": assay_type.get(train_id, "unknown"),
                "nearest_train_assay_name": assay_name.get(train_id, ""),
                "nearest_biolord_cosine": nearest_sim.get(test_id, np.nan),
                "feature": args.feature,
                "threshold": rule["threshold"],
                "direction": rule["direction"],
                "train_balanced_accuracy": rule["train_balanced_accuracy"],
                "train_binary_auroc": rule["train_binary_auroc"],
                "n_train_labels": int(len(y_train)),
                "n_train_pos": int((y_train == 1).sum()),
                "n_train_neg": int((y_train == 0).sum()),
                "test_set_length": int(len(y_test)),
                "num_actives": int((y_test == 1).sum()),
                "num_inactives": int((y_test == 0).sum()),
                "constant_fallback": not bool(rule["direction"]),
                **metrics,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "per_assay_metrics.csv", index=False)
    if logits:
        pd.concat(logits, ignore_index=True).to_csv(args.out_dir / "logits.csv", index=False)

    overall = pd.DataFrame([summarize(metrics)])
    overall.to_csv(args.out_dir / "summary_overall.csv", index=False)

    category_rows = []
    for assay_type_name, g in metrics.groupby("assay_type", sort=False):
        row = {"assay_type": assay_type_name}
        row.update(summarize(g))
        category_rows.append(row)
    by_category = pd.DataFrame(category_rows)
    by_category.to_csv(args.out_dir / "summary_by_category.csv", index=False)

    print("Overall summary:")
    print(overall.to_string(index=False))
    print("\nBy-category summary:")
    print(by_category.to_string(index=False))


if __name__ == "__main__":
    main()
