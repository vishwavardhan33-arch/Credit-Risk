"""Smoke tests on tiny synthetic CSVs, so they run without the Kaggle data."""
import numpy as np
import pandas as pd

from src.features.build_features import ID_COL, build_feature_table, to_model_frame


def _make_raw(tmp_path, n=50, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + n)

    app = pd.DataFrame({
        ID_COL: ids,
        "TARGET": rng.integers(0, 2, n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
        "AMT_INCOME_TOTAL": rng.uniform(0, 2e5, n),   # zeros possible -> tests safe_div
        "AMT_CREDIT": rng.uniform(1e4, 5e5, n),
        "AMT_ANNUITY": rng.uniform(1e3, 3e4, n),
        "DAYS_BIRTH": -rng.integers(7000, 25000, n),
        "DAYS_EMPLOYED": rng.choice([-1000, -200, 365243], n),
        "EXT_SOURCE_1": rng.uniform(0, 1, n),
        "EXT_SOURCE_2": rng.uniform(0, 1, n),
        "EXT_SOURCE_3": rng.uniform(0, 1, n),
    })
    app.loc[0, "AMT_INCOME_TOTAL"] = 0
    app.to_csv(tmp_path / "application_train.csv", index=False)

    def pick(m):  # rows spread over only some applicants, so some get no history
        return rng.choice(ids[: n - 10], m)

    m = 300
    pd.DataFrame({
        ID_COL: pick(m), "SK_ID_BUREAU": np.arange(m),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], m),
        "CREDIT_DAY_OVERDUE": rng.choice([0, 0, 5], m),
        "DAYS_CREDIT": -rng.integers(1, 3000, m),
        "AMT_CREDIT_SUM": rng.choice([0, 1e4, 5e4], m),
        "AMT_CREDIT_SUM_DEBT": rng.uniform(0, 3e4, m),
        "AMT_CREDIT_SUM_OVERDUE": rng.choice([0, 100], m),
    }).to_csv(tmp_path / "bureau.csv", index=False)

    pd.DataFrame({
        ID_COL: pick(m), "SK_ID_PREV": np.arange(m),
        "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused"], m),
        "AMT_ANNUITY": rng.uniform(1e3, 2e4, m),
        "AMT_APPLICATION": rng.uniform(0, 1e5, m),
        "AMT_CREDIT": rng.choice([0, 5e4, 9e4], m),
        "DAYS_DECISION": -rng.integers(1, 2000, m),
        "DAYS_FIRST_DRAWING": rng.choice([-300, 365243], m),
        "DAYS_FIRST_DUE": rng.choice([-250, 365243], m),
        "DAYS_LAST_DUE_1ST_VERSION": rng.choice([-100, 365243], m),
        "DAYS_LAST_DUE": rng.choice([-50, 365243], m),
        "DAYS_TERMINATION": rng.choice([-40, 365243], m),
    }).to_csv(tmp_path / "previous_application.csv", index=False)

    m = 800
    pd.DataFrame({
        ID_COL: pick(m), "SK_ID_PREV": rng.integers(0, 300, m),
        "DAYS_INSTALMENT": -rng.integers(1, 1000, m),
        "DAYS_ENTRY_PAYMENT": -rng.integers(1, 1000, m),
        "AMT_INSTALMENT": rng.choice([0, 1e3, 2e3], m),
        "AMT_PAYMENT": rng.uniform(0, 2e3, m),
    }).to_csv(tmp_path / "installments_payments.csv", index=False)

    pd.DataFrame({
        ID_COL: pick(m), "SK_ID_PREV": rng.integers(0, 300, m),
        "MONTHS_BALANCE": -rng.integers(1, 60, m),
        "SK_DPD": rng.choice([0, 0, 10], m), "SK_DPD_DEF": rng.choice([0, 0, 3], m),
    }).to_csv(tmp_path / "POS_CASH_balance.csv", index=False)

    pd.DataFrame({
        ID_COL: pick(m), "SK_ID_PREV": rng.integers(0, 300, m),
        "AMT_BALANCE": rng.uniform(0, 5e4, m),
        "AMT_CREDIT_LIMIT_ACTUAL": rng.choice([0, 1e4, 5e4], m),
        "AMT_DRAWINGS_CURRENT": rng.uniform(0, 1e4, m),
        "SK_DPD": rng.choice([0, 0, 7], m),
    }).to_csv(tmp_path / "credit_card_balance.csv", index=False)
    return len(ids)


def test_one_row_per_applicant_and_no_inf(tmp_path):
    n = _make_raw(tmp_path)
    full = build_feature_table(tmp_path, expected_rows=n)
    assert len(full) == n and full[ID_COL].is_unique
    num = full.select_dtypes("number")
    assert not np.isinf(num.to_numpy()).any()


def test_placeholder_and_no_leakage(tmp_path):
    _make_raw(tmp_path)
    full = build_feature_table(tmp_path)
    assert (full["DAYS_EMPLOYED"] == 365243).sum() == 0
    assert full["DAYS_EMPLOYED_ANOM"].sum() > 0
    assert not any("TARGET" in c for c in full.columns if c != "TARGET")


def test_include_extra_flag_and_model_frame(tmp_path):
    _make_raw(tmp_path)
    base = build_feature_table(tmp_path, include_extra=False)
    extra = build_feature_table(tmp_path, include_extra=True)
    assert not any(c.startswith(("POS_", "CC_")) for c in base.columns)
    assert any(c.startswith("POS_") for c in extra.columns)
    assert any(c.startswith("CC_") for c in extra.columns)
    X, y = to_model_frame(extra)
    assert "TARGET" not in X.columns and ID_COL not in X.columns and len(y) == len(X)
    assert str(X["NAME_CONTRACT_TYPE"].dtype) == "category"
