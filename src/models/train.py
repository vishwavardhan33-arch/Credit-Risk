"""Reproducible training pipeline: the script version of notebook 03, with MLflow tracking.

    python -m src.models.train --features data/processed/train_features.parquet --model-dir models
    python -m src.models.train ... --drop CODE_GENDER --run-name no-gender      # compare in MLflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db                            # browse runs

Steps: out-of-fold predictions -> pick calibrator -> cost-based threshold ->
fit final model -> one-time test evaluation -> save artifacts -> log to MLflow.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.calibration import apply_calibration, choose_calibrator, choose_threshold, total_profit

ID_COL = "SK_ID_CURR"
DEFAULT_LGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.7, random_state=42, verbose=-1)


def _metrics(p, y):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p))}


def run(features_path, model_dir, drop=(), lgd=0.60, margin=0.10, lgb_overrides=None,
        n_splits=5, seed=42, use_mlflow=True, mlflow_uri="sqlite:///mlflow.db",
        experiment="credit-risk", run_name=None, reference_rows=20000) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

    full = pd.read_parquet(features_path)
    missing = [c for c in drop if c not in full.columns]
    assert not missing, f"--drop columns not in the data: {missing}"

    X = full.drop(columns=["TARGET", ID_COL, *drop])
    y = full["TARGET"]
    credit = full["AMT_CREDIT"]            # taken from `full`, so it is available even if dropped from X
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("category")

    X_train, X_test, y_train, y_test, credit_train, credit_test = train_test_split(
        X, y, credit, test_size=0.2, stratify=y, random_state=seed)
    y_tr, y_te = y_train.to_numpy(), y_test.to_numpy()
    print(f"train {X_train.shape} | test {X_test.shape} | default rate {y_tr.mean():.4f}")

    params = {**DEFAULT_LGB_PARAMS, **(lgb_overrides or {})}

    # 1) out-of-fold predictions: every training row scored by a model that never saw it
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = cross_val_predict(LGBMClassifier(**params), X_train, y_train,
                            cv=cv, method="predict_proba")[:, 1]
    oof_metrics = _metrics(oof, y_tr)

    # 2) calibration, chosen on out-of-fold scores
    kind, cal_params, cv_cal = choose_calibrator(oof, y_tr)
    p_oof = cv_cal[kind]

    # 3) cost-based threshold, chosen on calibrated out-of-fold scores
    th = choose_threshold(p_oof, y_tr, credit_train.to_numpy(), lgd, margin)
    thr = th["threshold"]

    # 4) final model + one-time test evaluation
    model = LGBMClassifier(**params).fit(X_train, y_train)
    p_test = apply_calibration(model.predict_proba(X_test)[:, 1], cal_params)
    test_metrics = _metrics(p_test, y_te)
    approved = p_test < thr
    ct = credit_test.to_numpy()
    profit_thr = total_profit(p_test, y_te, ct, thr, lgd, margin)
    profit_all = total_profit(p_test, y_te, ct, 1.01, lgd, margin)
    test_summary = {
        "approval_rate": round(float(approved.mean()), 4),
        "default_rate_among_approved": round(float(y_te[approved].mean()), 4) if approved.any() else None,
        "default_rate_overall": round(float(y_te.mean()), 4),
        "defaults_rejected_pct": round(float((~approved & (y_te == 1)).sum() / max(1, (y_te == 1).sum())), 4),
        "profit_uplift_vs_approve_all_pct": round(100 * (profit_thr / profit_all - 1), 2) if profit_all > 0 else None,
    }

    # 5) artifacts
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "threshold": thr,
        "assumptions": {"LGD": lgd, "MARGIN": margin},
        "calibration": cal_params,
        "feature_columns": list(X_train.columns),
        "categorical_levels": {c: [str(v) for v in X_train[c].cat.categories] for c in cat_cols},
        "dropped_columns": list(drop),
        "lgb_params": params,
        "train_rows": int(len(X_train)),
        "oof_metrics": oof_metrics,
        "test_metrics": test_metrics,
        "test_summary": test_summary,
    }
    joblib.dump(model, model_dir / "lgbm_model.joblib")
    (model_dir / "model_config.json").write_text(json.dumps(config))

    # reference sample for drift monitoring: training rows with honest out-of-fold scores
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), size=min(reference_rows, len(X_train)), replace=False)
    ref = X_train.iloc[idx].copy()
    ref["score"] = p_oof[idx]
    ref.to_parquet(model_dir / "reference_sample.parquet")

    print(f"calibrator: {kind} | threshold: {thr:.3f} (break-even {th['analytic']:.3f})")
    print("OOF :", {k: round(v, 4) for k, v in oof_metrics.items()})
    print("test:", {k: round(v, 4) for k, v in test_metrics.items()})
    print("test summary:", test_summary)

    if use_mlflow:
        _log_mlflow(config, params, kind, oof_metrics, test_metrics, test_summary, model_dir,
                    mlflow_uri, experiment, run_name, X_train.shape)
    return config


def _log_mlflow(config, params, kind, oof_metrics, test_metrics, test_summary, model_dir,
                uri, experiment, run_name, train_shape):
    import mlflow
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({f"lgb_{k}": v for k, v in params.items() if k != "verbose"})
        mlflow.log_params({
            "calibrator": kind,
            "LGD": config["assumptions"]["LGD"], "MARGIN": config["assumptions"]["MARGIN"],
            "dropped_columns": ",".join(config["dropped_columns"]) or "none",
            "n_train": train_shape[0], "n_features": train_shape[1],
        })
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({"threshold": config["threshold"]})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_summary.items() if v is not None})
        mlflow.set_tag("model_version", config["model_version"])
        mlflow.log_artifact(str(model_dir / "model_config.json"))
        mlflow.log_artifact(str(model_dir / "lgbm_model.joblib"))
    print(f"logged to MLflow experiment '{experiment}' at {uri}")


def main():
    ap = argparse.ArgumentParser(description="Train, calibrate, set threshold, save, log to MLflow.")
    ap.add_argument("--features", default="data/processed/train_features.parquet")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--drop", nargs="*", default=[], help="feature columns to exclude, e.g. CODE_GENDER")
    ap.add_argument("--lgd", type=float, default=0.60)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--experiment", default="credit-risk")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    run(a.features, a.model_dir, drop=a.drop, lgd=a.lgd, margin=a.margin, use_mlflow=not a.no_mlflow,
        mlflow_uri=a.mlflow_uri, experiment=a.experiment, run_name=a.run_name)


if __name__ == "__main__":
    main()
