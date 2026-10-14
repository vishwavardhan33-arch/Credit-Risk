import json

import pytest

from src.models.train import run


def test_artifacts_and_config(artifacts):
    d = artifacts["model_dir"]
    assert {"lgbm_model.joblib", "model_config.json", "reference_sample.parquet"} <= {p.name for p in d.iterdir()}
    cfg = json.loads((d / "model_config.json").read_text())
    assert 0 < cfg["threshold"] < 1 and cfg["calibration"]["kind"] in ("sigmoid", "isotonic")
    assert cfg["test_metrics"]["roc_auc"] > 0.5            # synthetic data has real signal
    assert "TARGET" not in cfg["feature_columns"] and "SK_ID_CURR" not in cfg["feature_columns"]


def test_drop_column_and_mlflow_logging(artifacts, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    cfg = run(artifacts["features"], tmp_path / "m", drop=["CODE_GENDER"], lgb_overrides={"n_estimators": 20},
              n_splits=3, mlflow_uri=uri, experiment="t", run_name="no-gender")
    assert "CODE_GENDER" not in cfg["feature_columns"] and cfg["dropped_columns"] == ["CODE_GENDER"]
    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["t"])
    assert len(runs) == 1
    assert runs.loc[0, "params.dropped_columns"] == "CODE_GENDER"
    assert runs.loc[0, "metrics.test_roc_auc"] > 0
