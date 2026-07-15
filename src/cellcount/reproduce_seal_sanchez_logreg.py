"""Reproduce the Cell Count paper's Sanchez/CLOOME cell-count linear probe."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src/cellcount/resources/The_Sanchez-Fernandez_Files"
OUT_DIR = ROOT / "src/cellcount/results/reproduce_seal_sanchez_logreg"


EXCLUDED_COLUMNS = [
    "INCHIKEY",
    "Unnamed: 0",
    "Cells_Number_Object_Number",
    "Cells_Neighbors_FirstClosestObjectNumber_5",
    "Cells_Neighbors_FirstClosestObjectNumber_Adjacent",
    "Cells_Neighbors_SecondClosestObjectNumber_5",
    "Cells_Neighbors_SecondClosestObjectNumber_Adjacent",
    "Cells_Parent_Nuclei",
    "Cytoplasm_Number_Object_Number",
    "Cytoplasm_Parent_Cells",
    "Cytoplasm_Parent_Nuclei",
    "Nuclei_Neighbors_FirstClosestObjectNumber_1",
    "Nuclei_Neighbors_SecondClosestObjectNumber_1",
    "Nuclei_Number_Object_Number",
    "InChIKey",
]


def merge_data(data: pd.DataFrame, cp_data: pd.DataFrame) -> pd.DataFrame:
    return data.merge(cp_data, on="INCHIKEY")


def perform_logistic_regression(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    assay: str,
    train_cols: list[str],
) -> tuple[float | None, float | None, float | None]:
    x_train = train[train_cols]
    y_train = train[assay]
    x_val = val[train_cols]
    y_val = val[assay]
    x_test = test[train_cols]
    y_test = test[assay]

    if (
        len(y_train) == 0
        or len(y_val) == 0
        or len(y_test) == 0
        or len(y_train.unique()) == 1
        or len(y_val.unique()) == 1
        or len(y_test.unique()) == 1
    ):
        return None, None, None

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=train_cols)
    x_val = pd.DataFrame(scaler.transform(x_val), columns=train_cols)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=train_cols)

    best_auc = 0
    best_c = None
    best_model = LogisticRegression(C=1, max_iter=3000, random_state=42)
    best_model.fit(x_train, y_train)

    for c_value in [10**i for i in range(-6, 7)]:
        model = LogisticRegression(C=c_value, max_iter=3000, random_state=42)
        model.fit(x_train, y_train)
        val_auc = roc_auc_score(y_val, model.predict_proba(x_val)[:, 1])
        if val_auc > best_auc:
            best_auc = val_auc
            best_c = c_value
            best_model = model

    x_train_val = pd.concat([x_train, x_val])
    y_train_val = pd.concat([y_train, y_val])
    best_model.fit(x_train_val, y_train_val)
    test_auc = roc_auc_score(y_test, best_model.predict_proba(x_test)[:, 1])
    return best_c, best_auc, test_auc


def process_split(
    cp_data: pd.DataFrame,
    assay_columns: list[str],
    train_cols: list[str],
    split_idx: int,
) -> pd.DataFrame:
    train = pd.read_csv(DATA_DIR / f"data/datasplit{split_idx}-train.csv")
    val = pd.read_csv(DATA_DIR / f"data/datasplit{split_idx}-val.csv")
    test = pd.read_csv(DATA_DIR / f"data/datasplit{split_idx}-test.csv")

    train_merged = merge_data(train, cp_data)
    val_merged = merge_data(val, cp_data)
    test_merged = merge_data(test, cp_data)

    rows = []
    for assay in tqdm(assay_columns, desc=f"split{split_idx}"):
        train_assay = train_merged.dropna(subset=[assay])[[assay] + train_cols]
        val_assay = val_merged.dropna(subset=[assay])[[assay] + train_cols]
        test_assay = test_merged.dropna(subset=[assay])[[assay] + train_cols]
        best_c, val_auc, test_auc = perform_logistic_regression(
            train_assay, val_assay, test_assay, assay, train_cols
        )
        if best_c is not None:
            rows.append(
                {
                    "assay": assay,
                    "best_C": best_c,
                    "val_auc": val_auc,
                    "test_auc": test_auc,
                    "split": f"split{split_idx}",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    cp_data = pd.read_csv(DATA_DIR / "CP_count_Sanchez.csv")
    assay_columns = [col for col in cp_data.columns if col not in EXCLUDED_COLUMNS]
    cp_data[assay_columns] = cp_data[assay_columns].replace(-1, np.nan)
    train_cols = ["Cells_Number_Object_Number"]

    out = []
    for split_idx in [1, 2, 3]:
        out.append(process_split(cp_data, assay_columns, train_cols, split_idx))
    all_results = pd.concat(out, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(OUT_DIR / "logistic_regression_results_CellCount.csv", index=False)

    assay_mean = all_results.groupby("assay", as_index=False)["test_auc"].mean()
    assay_mean.to_csv(OUT_DIR / "assay_mean_test_auc.csv", index=False)
    summary = {
        "fold_rows": int(len(all_results)),
        "n_assays": int(len(assay_mean)),
        "mean_auroc": float(assay_mean["test_auc"].mean()),
        "std_auroc": float(assay_mean["test_auc"].std()),
        "auroc_ge_0.9": int((assay_mean["test_auc"] >= 0.9).sum()),
        "auroc_ge_0.8": int((assay_mean["test_auc"] >= 0.8).sum()),
        "auroc_ge_0.7": int((assay_mean["test_auc"] >= 0.7).sum()),
        "auroc_ge_0.5": int((assay_mean["test_auc"] >= 0.5).sum()),
    }
    pd.Series(summary).to_json(OUT_DIR / "summary.json", indent=2)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
