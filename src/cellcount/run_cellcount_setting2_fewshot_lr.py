#!/usr/bin/env python
"""Per-assay logistic-regression cell-count baseline for Setting 2 few-shot."""

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
FRACTIONS = {
    "0.1%": "K10",
    "1%": "K100",
    "10%": "K1000",
    "100%": "K10000",
}


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
        "--assay-split",
        type=Path,
        default=PROJECT_ROOT / "src/data/Broad-270/assay_splits/fewshotold/label_test.csv",
    )
    parser.add_argument(
        "--fewshot-dir",
        type=Path,
        default=PROJECT_ROOT / "src/data/Broad-270/assay_splits/fewshotold",
    )
    parser.add_argument(
        "--assay-meta",
        type=Path,
        default=PROJECT_ROOT / "src/data/Broad-270/assay_meta/assay_meta.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "src/cellcount/results/setting2_fewshot_curve_trainonly_cellcount_lr",
    )
    parser.add_argument("--feature", default="Cells_Number_Object_Number")
    return parser.parse_args()


def read_assay_ids(path: Path) -> list[str]:
    return [c for c in pd.read_csv(path, nrows=0).columns if c != "smiles"]


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


def summarize(df: pd.DataFrame) -> dict[str, float]:
    valid = df.dropna(subset=["auroc"])
    filled = df["auroc"].fillna(0.5)
    return {
        "n_assays": int(len(df)),
        "n_valid_assays": int(len(valid)),
        "n_constant_fallback_assays": int(df["auroc"].isna().sum()),
        "mean_auroc_valid": float(valid["auroc"].mean()) if len(valid) else float("nan"),
        "median_auroc_valid": float(valid["auroc"].median()) if len(valid) else float("nan"),
        "mean_auprc_valid": float(valid["auprc"].mean()) if len(valid) else float("nan"),
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

    assays = read_assay_ids(args.assay_split)
    labels = pd.read_csv(args.label_csv)
    cellcount = pd.read_csv(args.cellcount_csv)
    meta = pd.read_csv(args.assay_meta, dtype={"PUMA_ASSAY_ID": str})
    assay_type = meta.set_index("PUMA_ASSAY_ID")["ASSAY_TYPE"].to_dict()

    data = labels[["smiles", *assays]].merge(
        cellcount[["smiles", args.feature]],
        on="smiles",
        how="left",
    )
    if data[args.feature].isna().any():
        raise ValueError("Some compounds have no cell-count feature")

    all_rows = []
    all_logits = []
    for fraction, split_name in FRACTIONS.items():
        split = pd.read_csv(args.fewshot_dir / f"train_test_split_{split_name}.csv")
        train_smiles = set(split["Train_SMILES"].dropna().astype(str))
        test_smiles = set(split["Test_SMILES"].dropna().astype(str))
        valid_smiles = set(split["Valid_SMILES"].dropna().astype(str))
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
                "fraction": fraction,
                "split_name": split_name,
                "assay_id": assay,
                "assay_type": assay_type.get(assay, "unknown"),
                "auroc": np.nan,
                "auprc": np.nan,
                "test_set_length": int(len(y_test)),
                "num_actives": int((y_test == 1).sum()),
                "num_inactives": int((y_test == 0).sum()),
                "n_train_compounds": int(len(train_smiles)),
                "n_valid_compounds_unused": int(len(valid_smiles)),
                "n_test_compounds": int(len(test_smiles)),
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
                logits = test_assay[["smiles", "label"]].copy()
                logits["fraction"] = fraction
                logits["split_name"] = split_name
                logits["assay_id"] = assay
                logits["score"] = score
                all_logits.append(logits[["fraction", "split_name", "assay_id", "smiles", "score", "label"]])
            all_rows.append(row)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(args.out_dir / "per_assay_metrics_lr.csv", index=False)
    if all_logits:
        pd.concat(all_logits, ignore_index=True).to_csv(args.out_dir / "logits_lr.csv", index=False)

    overall_rows = []
    for fraction, group in metrics.groupby("fraction", sort=False):
        row = {"fraction": fraction}
        row.update(summarize(group))
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(args.out_dir / "summary_overall_lr.csv", index=False)

    category_rows = []
    for (fraction, assay_type_name), group in metrics.groupby(["fraction", "assay_type"], sort=False):
        row = {"fraction": fraction, "assay_type": assay_type_name}
        row.update(summarize(group))
        category_rows.append(row)
    by_category = pd.DataFrame(category_rows)
    by_category.to_csv(args.out_dir / "summary_by_category_lr.csv", index=False)
    summary = {"overall": overall.to_dict(orient="records"), "by_category": by_category.to_dict(orient="records")}
    (args.out_dir / "summary_lr.json").write_text(json.dumps(summary, indent=2))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
