"""Shared fixture: a small synthetic feature table and a model trained on it with src.models.train."""
import numpy as np
import pytest

from src.features.build_features import build_feature_table
from src.models.train import run
from tests.test_features import _make_raw


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    _make_raw(raw, n=6000)
    full = build_feature_table(raw)
    rng = np.random.default_rng(1)
    signal = -2.4 - 3.0 * (full["EXT_MEAN"].fillna(0.5) - 0.5)
    full["TARGET"] = (rng.random(len(full)) < 1 / (1 + np.exp(-signal))).astype(int)
    full["CODE_GENDER"] = rng.choice(["F", "M"], len(full))
    features = raw / "train_features.parquet"
    full.to_parquet(features)

    model_dir = tmp_path_factory.mktemp("models")
    run(features, model_dir, lgb_overrides={"n_estimators": 40}, n_splits=3, use_mlflow=False)
    return {"features": features, "model_dir": model_dir, "full": full}
