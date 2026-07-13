#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


BROAD270_ALL_F1_POSITIVE_RATE = 0.027209
BROAD270_TRAIN_ASSAY_F1_POSITIVE_RATE = 0.025989
CHEMBL209_F1_POSITIVE_RATE = 0.347017
EVE24_EXTERNAL_F1_POSITIVE_RATE = 0.027209


def auroc_or_default(labels: np.ndarray, scores: np.ndarray, default: float | None = None) -> float:
    labels = labels.astype(int)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan") if default is None else float(default)
    return float(roc_auc_score(labels, scores))


def auprc_or_default(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    if len(labels) == 0:
        return float("nan")
    if labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, scores))


def f1_at_top_fraction(labels: np.ndarray, scores: np.ndarray, positive_rate: float) -> float:
    labels = labels.astype(int)
    if len(labels) == 0:
        return float("nan")
    n_pred_pos = int(round(len(labels) * positive_rate))
    n_pred_pos = min(max(n_pred_pos, 1), len(labels))
    order = np.argsort(-scores, kind="mergesort")
    pred = np.zeros(len(labels), dtype=int)
    pred[order[:n_pred_pos]] = 1
    return float(f1_score(labels, pred, zero_division=0))


def enrichment_factor(labels: np.ndarray, scores: np.ndarray, top_fraction: float = 0.05) -> float:
    labels = labels.astype(int)
    if len(labels) == 0:
        return float("nan")
    base_rate = labels.mean()
    if base_rate <= 0:
        return float("nan")
    top_n = int(np.ceil(len(labels) * top_fraction))
    top_n = min(max(top_n, 1), len(labels))
    order = np.argsort(-scores, kind="mergesort")
    return float(labels[order[:top_n]].mean() / base_rate)


def bootstrap_ci(values: Iterable[float], n_bootstrap: int = 1000, seed: int = 0) -> tuple[float, float]:
    vals = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        boot[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def infer_f1_positive_rate(path: Path, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "eve" in name:
        return EVE24_EXTERNAL_F1_POSITIVE_RATE
    if "setting1_openpheno_209" in name:
        return CHEMBL209_F1_POSITIVE_RATE
    if "setting1_openpheno_270" in name:
        return BROAD270_ALL_F1_POSITIVE_RATE
    if parent == "fewshot" or "setting2" in name or "setting3_broad" in name:
        return BROAD270_TRAIN_ASSAY_F1_POSITIVE_RATE
    return BROAD270_TRAIN_ASSAY_F1_POSITIVE_RATE


def compute_per_assay_metrics(
    path: Path,
    f1_positive_rate: float,
    invalid_auroc_default: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"assay_id", "prob", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    rows = []
    for assay_id, g in df.groupby("assay_id", sort=True):
        labels = g["label"].to_numpy(dtype=int)
        scores = g["prob"].to_numpy(dtype=float)
        rows.append(
            {
                "assay_id": assay_id,
                "n": int(len(labels)),
                "positives": int(labels.sum()),
                "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
                "AUROC": auroc_or_default(labels, scores, default=invalid_auroc_default),
                "AUPRC": auprc_or_default(labels, scores),
                "F1": f1_at_top_fraction(labels, scores, f1_positive_rate),
                "EF5": enrichment_factor(labels, scores, top_fraction=0.05),
            }
        )
    return pd.DataFrame(rows)


def summarize_per_assay(per_assay: pd.DataFrame, dataset: str, method: str, source_file: str) -> dict:
    row = {
        "dataset": dataset,
        "method": method,
        "source_file": source_file,
        "num_assays": int(len(per_assay)),
    }
    for metric in ["AUROC", "AUPRC", "F1", "EF5"]:
        vals = per_assay[metric].dropna().to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals)
        row[f"mean_{metric}"] = float(vals.mean()) if len(vals) else float("nan")
        row[f"{metric}_ci_low"] = lo
        row[f"{metric}_ci_high"] = hi
    if "AUROC" in per_assay:
        row["AUROC_gt_0.9"] = int((per_assay["AUROC"] > 0.9).sum())
        row["AUROC_gt_0.7"] = int((per_assay["AUROC"] > 0.7).sum())
        row["AUROC_gt_0.5"] = int((per_assay["AUROC"] > 0.5).sum())
    return row


def parse_result_name(path: Path) -> tuple[str, str]:
    name = path.stem
    if name.startswith("setting1_OpenPheno_270"):
        return "Broad-270", "OpenPheno"
    if name.startswith("setting1_OpenPheno_209"):
        return "ChEMBL-209", "OpenPheno"
    if name.startswith("setting2_"):
        return "Broad-270 Setting 2", name.replace("setting2_", "")
    if name.startswith("setting3_Broad"):
        return "Broad-270 Setting 3", "OpenPheno"
    if name.startswith("setting3_Eve"):
        return "EvE-24 Setting 3", "OpenPheno"
    if path.parent.name == "fewshot":
        return "Broad-270 Few-shot", name.replace("logits", "")
    return "Unknown", name


def setting1_fold_id(path: Path) -> str:
    match = re.search(r"_(270|209)(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse fold id from {path.name}")
    return match.group(2)


def summarize_setting1_broad270(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for path in sorted(paths, key=setting1_fold_id):
        fold = setting1_fold_id(path)
        per = compute_per_assay_metrics(path, BROAD270_ALL_F1_POSITIVE_RATE)
        per["fold"] = fold
        fold_rows.append(per)
    all_folds = pd.concat(fold_rows, ignore_index=True)
    per_assay = (
        all_folds.groupby("assay_id", as_index=False)
        .agg(
            n=("n", "median"),
            positives=("positives", "median"),
            positive_rate=("positive_rate", "median"),
            AUROC=("AUROC", "median"),
            AUPRC=("AUPRC", "median"),
            F1=("F1", "median"),
            EF5=("EF5", "median"),
        )
    )
    summary = pd.DataFrame([summarize_per_assay(per_assay, "Broad-270", "OpenPheno", "setting1_OpenPheno_270[1-5].csv")])
    summary["aggregation"] = "median across five folds per assay, then mean across assays"
    return summary, per_assay


def summarize_setting1_chembl209(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    for path in sorted(paths, key=setting1_fold_id):
        fold = setting1_fold_id(path)
        per = compute_per_assay_metrics(path, CHEMBL209_F1_POSITIVE_RATE, invalid_auroc_default=0.5)
        per["fold"] = fold
        fold_rows.append(per)
    all_folds = pd.concat(fold_rows, ignore_index=True)
    assay_ids = sorted(all_folds["assay_id"].unique())
    folds = sorted(all_folds["fold"].unique())
    completed = []
    for assay_id in assay_ids:
        subset = all_folds[all_folds["assay_id"] == assay_id].copy()
        observed = set(subset["fold"])
        template = subset.iloc[0].to_dict()
        for fold in folds:
            if fold not in observed:
                row = template.copy()
                row.update({"fold": fold, "AUROC": 0.5, "AUPRC": np.nan, "F1": np.nan, "EF5": np.nan})
                completed.append(row)
        completed.extend(subset.to_dict("records"))
    all_completed = pd.DataFrame(completed)
    per_assay = (
        all_completed.groupby("assay_id", as_index=False)
        .agg(
            n=("n", "median"),
            positives=("positives", "median"),
            positive_rate=("positive_rate", "median"),
            AUROC=("AUROC", "mean"),
            AUPRC=("AUPRC", "mean"),
            F1=("F1", "mean"),
            EF5=("EF5", "mean"),
        )
    )
    summary = pd.DataFrame([summarize_per_assay(per_assay, "ChEMBL-209", "OpenPheno", "setting1_OpenPheno_209[1-3].csv")])
    summary["aggregation"] = "mean across three folds per assay with missing/invalid AUROC filled as 0.5, then mean across assays"
    return summary, per_assay


def discover_files(results_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in results_dir.rglob("*.csv")
        if not p.name.startswith("metrics_")
        and not p.name.startswith("per_assay_")
        and not p.name.startswith("summary_")
    )


def calculate_all(results_dir: Path, out_dir: Path, f1_positive_rate: float | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = discover_files(results_dir)
    summary_rows = []

    broad270_files = [p for p in files if p.name.startswith("setting1_OpenPheno_270")]
    chembl209_files = [p for p in files if p.name.startswith("setting1_OpenPheno_209")]

    skip = set(broad270_files + chembl209_files)
    if broad270_files:
        summary, per_assay = summarize_setting1_broad270(broad270_files)
        summary_rows.extend(summary.to_dict("records"))
        per_assay.to_csv(out_dir / "per_assay_setting1_Broad270.csv", index=False)
    if chembl209_files:
        summary, per_assay = summarize_setting1_chembl209(chembl209_files)
        summary_rows.extend(summary.to_dict("records"))
        per_assay.to_csv(out_dir / "per_assay_setting1_ChEMBL209.csv", index=False)

    for path in files:
        if path in skip:
            continue
        dataset, method = parse_result_name(path)
        rate = infer_f1_positive_rate(path, override=f1_positive_rate)
        per_assay = compute_per_assay_metrics(path, rate)
        safe_name = path.relative_to(results_dir).with_suffix("").as_posix().replace("/", "_")
        per_assay.to_csv(out_dir / f"per_assay_{safe_name}.csv", index=False)
        row = summarize_per_assay(per_assay, dataset, method, path.relative_to(results_dir).as_posix())
        row["f1_positive_rate"] = rate
        row["aggregation"] = "direct mean across assays"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {out_dir / 'summary_metrics.csv'}")
    print(f"Saved per-assay metric files to {out_dir}")


def calculate_one(path: Path, out_dir: Path, f1_positive_rate: float | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset, method = parse_result_name(path)
    rate = infer_f1_positive_rate(path, override=f1_positive_rate)
    per_assay = compute_per_assay_metrics(path, rate)
    summary = pd.DataFrame([summarize_per_assay(per_assay, dataset, method, path.name)])
    summary["f1_positive_rate"] = rate
    per_assay.to_csv(out_dir / f"per_assay_{path.stem}.csv", index=False)
    summary.to_csv(out_dir / f"summary_{path.stem}.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate OpenPheno AUROC, AUPRC, F1, EF@5%, and 95% bootstrap CIs.")
    parser.add_argument("--results_dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--f1_positive_rate", type=float, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.results_dir / "metrics")
    if args.file is not None:
        calculate_one(args.file, out_dir, args.f1_positive_rate)
    else:
        calculate_all(args.results_dir, out_dir, args.f1_positive_rate)


if __name__ == "__main__":
    main()
