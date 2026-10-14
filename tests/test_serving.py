import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.scoring import Scorer


def _records(full, n, scorer):
    cols = scorer.columns + ["SK_ID_CURR"]
    return json.loads(full[cols].head(n).to_json(orient="records"))


def test_health_and_info(artifacts):
    with TestClient(create_app(str(artifacts["model_dir"]))) as client:
        assert client.get("/health").json()["status"] == "ok"
        info = client.get("/model/info").json()
        assert 0 < info["threshold"] < 1 and info["n_features"] > 10


def test_predict_matches_direct_scoring(artifacts):
    scorer = Scorer(artifacts["model_dir"])
    recs = _records(artifacts["full"], 8, scorer)
    with TestClient(create_app(str(artifacts["model_dir"]))) as client:
        r = client.post("/predict", json={"applications": recs, "explain_top_k": 3})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    direct = scorer.score_frame(pd.DataFrame(recs))
    assert np.allclose([p["default_probability"] for p in preds], direct, atol=1e-5)
    for p in preds:
        assert p["decision"] == ("approve" if p["default_probability"] < p["threshold"] else "reject")
        assert len(p["top_risk_factors"]) <= 3
        assert all(f["contribution"] > 0 for f in p["top_risk_factors"])
    assert preds[0]["id"] == recs[0]["SK_ID_CURR"]


def test_warnings_missing_values_and_validation(artifacts):
    scorer = Scorer(artifacts["model_dir"])
    rec = _records(artifacts["full"], 1, scorer)[0]
    rec["NAME_CONTRACT_TYPE"] = "Spaceship loan"      # unseen category
    rec["TARGET"] = 1                                  # not a feature
    with TestClient(create_app(str(artifacts["model_dir"]))) as client:
        r = client.post("/predict", json={"applications": [rec], "explain_top_k": 0})
        assert r.status_code == 200
        p = r.json()["predictions"][0]
        assert any("unseen category" in w for w in p["warnings"])
        assert any("unknown fields" in w for w in p["warnings"])
        assert p["top_risk_factors"] == []

        empty = client.post("/predict", json={"applications": [{}]})        # everything missing
        assert empty.status_code == 200 and 0 <= empty.json()["predictions"][0]["default_probability"] <= 1
        assert empty.json()["predictions"][0]["missing_features"] == len(scorer.columns)

        assert client.post("/predict", json={"applications": []}).status_code == 422
        assert client.post("/predict", json={"applications": [{}], "explain_top_k": 99}).status_code == 422


def test_prediction_log_feeds_drift_loader(artifacts, tmp_path):
    from src.monitoring.drift_report import load_current
    scorer = Scorer(artifacts["model_dir"])
    log = tmp_path / "pred.jsonl"
    recs = _records(artifacts["full"], 5, scorer)
    with TestClient(create_app(str(artifacts["model_dir"]), prediction_log=str(log))) as client:
        client.post("/predict", json={"applications": recs[:3]})
        client.post("/predict", json={"applications": recs[3:]})
    df = load_current(log)
    assert len(df) == 5 and "score" in df.columns and set(scorer.columns) <= set(df.columns)
