"""
Feature engineering for the Home Credit Default Risk project.

Usage (from the credit-risk/ folder):
    python -m src.features.build_features                 # train set
    python -m src.features.build_features --split test    # test set
    python -m src.features.build_features --no-extra      # skip POS_CASH / credit_card tables

In a notebook:
    from src.features.build_features import build_feature_table
    full = build_feature_table("data/raw")

Every secondary table is aggregated to one row per SK_ID_CURR and left-joined
onto the application table, so the output has exactly one row per applicant.
No feature uses TARGET, so there is no target leakage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "SK_ID_CURR"
DAYS_PLACEHOLDER = 365243  # Home Credit uses this value to mean "not applicable"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def agg_table(df: pd.DataFrame, group_col: str, aggs: dict, prefix: str) -> pd.DataFrame:
    """Group by `group_col`, aggregate, and flatten to PREFIX_COLUMN_FUNC names."""
    out = df.groupby(group_col).agg(aggs)
    out.columns = [f"{prefix}_{col}_{func}".upper() for col, func in out.columns]
    return out.reset_index()


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """a / b with zero denominators turned into NaN (never inf)."""
    return a / b.replace(0, np.nan)


def _read(raw_dir: Path, name: str, usecols: list[str]) -> pd.DataFrame:
    """Read only the columns we need; this keeps the 13M-row installments table light."""
    return pd.read_csv(raw_dir / name, usecols=usecols)


# --------------------------------------------------------------------------- #
# application table
# --------------------------------------------------------------------------- #
def load_application(raw_dir: Path, split: str = "train") -> pd.DataFrame:
    df = pd.read_csv(raw_dir / f"application_{split}.csv")

    # 365243 in DAYS_EMPLOYED means "not employed", not ~1000 years of work
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_PLACEHOLDER, np.nan)

    df["CREDIT_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TERM"] = safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["EMPLOYED_AGE_RATIO"] = safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_MEAN"] = df[ext].mean(axis=1)
    df["EXT_PROD"] = df[ext].prod(axis=1)
    return df


# --------------------------------------------------------------------------- #
# secondary tables (each returns one row per SK_ID_CURR)
# --------------------------------------------------------------------------- #
def bureau_features(raw_dir: Path) -> pd.DataFrame:
    b = _read(raw_dir, "bureau.csv", [
        ID_COL, "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_DAY_OVERDUE", "DAYS_CREDIT",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    ])
    b["ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["OVERDUE_FLAG"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    out = agg_table(b, ID_COL, {
        "SK_ID_BUREAU": ["count"],
        "ACTIVE": ["mean", "sum"],
        "OVERDUE_FLAG": ["mean"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum"],
        "DAYS_CREDIT": ["min", "max", "mean"],
    }, "BUREAU")
    out["BUREAU_DEBT_CREDIT_RATIO"] = safe_div(
        out["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"], out["BUREAU_AMT_CREDIT_SUM_SUM"]
    )
    return out


def previous_features(raw_dir: Path) -> pd.DataFrame:
    day_cols = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                "DAYS_LAST_DUE", "DAYS_TERMINATION"]
    p = _read(raw_dir, "previous_application.csv", [
        ID_COL, "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_ANNUITY", "AMT_APPLICATION",
        "AMT_CREDIT", "DAYS_DECISION", *day_cols,
    ])
    for c in day_cols:
        p[c] = p[c].replace(DAYS_PLACEHOLDER, np.nan)
    p["REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["APP_CREDIT_RATIO"] = safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"])

    return agg_table(p, ID_COL, {
        "SK_ID_PREV": ["count"],
        "REFUSED": ["mean", "sum"],
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_CREDIT": ["sum", "mean"],
        "APP_CREDIT_RATIO": ["mean"],
        "DAYS_DECISION": ["min", "max"],
    }, "PREV")


def installments_features(raw_dir: Path) -> pd.DataFrame:
    i = _read(raw_dir, "installments_payments.csv", [
        ID_COL, "SK_ID_PREV", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT", "AMT_PAYMENT",
    ])
    i["DAYS_LATE"] = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]
    i["LATE_FLAG"] = (i["DAYS_LATE"] > 0).astype(int)
    i["UNDERPAID"] = (i["AMT_PAYMENT"] < i["AMT_INSTALMENT"] - 0.01).astype(int)
    i["PAY_RATIO"] = safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])

    return agg_table(i, ID_COL, {
        "SK_ID_PREV": ["nunique"],
        "DAYS_LATE": ["mean", "max"],
        "LATE_FLAG": ["mean", "sum"],
        "UNDERPAID": ["mean", "sum"],
        "PAY_RATIO": ["mean", "min"],
    }, "INST")


def pos_cash_features(raw_dir: Path) -> pd.DataFrame:
    p = _read(raw_dir, "POS_CASH_balance.csv", [
        ID_COL, "SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD", "SK_DPD_DEF",
    ])
    return agg_table(p, ID_COL, {
        "SK_ID_PREV": ["nunique"],
        "MONTHS_BALANCE": ["min", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }, "POS")


def credit_card_features(raw_dir: Path) -> pd.DataFrame:
    c = _read(raw_dir, "credit_card_balance.csv", [
        ID_COL, "SK_ID_PREV", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_CURRENT", "SK_DPD",
    ])
    c["UTILIZATION"] = safe_div(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    return agg_table(c, ID_COL, {
        "SK_ID_PREV": ["nunique"],
        "UTILIZATION": ["mean", "max"],
        "SK_DPD": ["max", "mean"],
        "AMT_BALANCE": ["mean", "max"],
        "AMT_DRAWINGS_CURRENT": ["sum", "mean"],
    }, "CC")


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_feature_table(
    raw_dir: str | Path,
    split: str = "train",
    include_extra: bool = True,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Return one row per applicant with application + aggregated history features.

    expected_rows: optional sanity check (307511 for train, 48744 for test).
    """
    raw_dir = Path(raw_dir)
    full = load_application(raw_dir, split)
    n_app = len(full)
    if expected_rows is not None:
        assert n_app == expected_rows, f"application has {n_app} rows, expected {expected_rows}"
    assert full[ID_COL].is_unique, "SK_ID_CURR must be unique in the application table"

    tables = [bureau_features, previous_features, installments_features]
    if include_extra:
        tables += [pos_cash_features, credit_card_features]

    for fn in tables:
        agg = fn(raw_dir)
        assert agg[ID_COL].is_unique, f"{fn.__name__} produced duplicate {ID_COL}"
        full = full.merge(agg, on=ID_COL, how="left")
        assert len(full) == n_app, f"row count changed after {fn.__name__}: {len(full)} != {n_app}"

    # belt and braces: ratios are already zero-safe, but make sure no inf survives
    num_cols = full.select_dtypes("number").columns
    full[num_cols] = full[num_cols].replace([np.inf, -np.inf], np.nan)
    return full


def to_model_frame(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Split into (X, y). Object columns become `category` for LightGBM. y is None for test."""
    X = full.drop(columns=[c for c in ["TARGET", ID_COL] if c in full.columns])
    for c in X.select_dtypes(include=["object", "string"]).columns:
        X[c] = X[c].astype("category")
    y = full["TARGET"] if "TARGET" in full.columns else None
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Home Credit feature table.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--no-extra", action="store_true", help="skip POS_CASH and credit_card tables")
    args = parser.parse_args()

    expected = {"train": 307511, "test": 48744}[args.split]
    full = build_feature_table(args.raw_dir, args.split, not args.no_extra, expected_rows=expected)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_features.parquet"
    full.to_parquet(out_path)
    print(f"{args.split}: {full.shape} -> {out_path}")


if __name__ == "__main__":
    main()
