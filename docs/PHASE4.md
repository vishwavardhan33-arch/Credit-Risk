# Phase 4: serving, tracking, monitoring

What was added and how to run it. Tested with the pytest suite and a live uvicorn run on synthetic data. The Dockerfile was **not** built or run (no Docker where it was written), so treat its first build as the real test.

| File | Purpose |
|---|---|
| `src/models/calibration.py` | calibrator and cost-based threshold helpers, shared by training and serving |
| `src/models/train.py` | reproducible training script (notebook 03 as code) with MLflow tracking |
| `src/serving/scoring.py`, `app.py` | scoring logic and the FastAPI service |
| `src/monitoring/drift_report.py` | PSI drift report on top features and on the predicted score |
| `docker/Dockerfile`, `requirements-serving.txt` | API image |
| `tests/` | 11 tests covering features, training, API and drift |

## 1. Train and track (once per model version)

```bash
python -m src.models.train --features data/processed/train_features.parquet --model-dir models
python -m src.models.train --features data/processed/train_features.parquet --model-dir models_nogender \
       --drop CODE_GENDER --run-name no-gender
mlflow ui --backend-store-uri sqlite:///mlflow.db        # compare runs at http://localhost:5000
```

This writes `lgbm_model.joblib`, `model_config.json` (threshold, calibrator, feature list, category levels) and `reference_sample.parquet` (training rows with out-of-fold scores, used as the drift baseline). Each run logs its parameters, out-of-fold and test metrics, threshold, and artifacts to MLflow. Use `--lgd` and `--margin` to set your real cost assumptions.

In Colab: `%cd /content/drive/MyDrive/credit-risk`, then run the same command with `!python -m src.models.train ...` (add `--mlflow-uri sqlite:////content/drive/MyDrive/credit-risk/mlflow.db` so the runs survive a runtime reset).

## 2. Run the API

```bash
MODEL_DIR=models PREDICTION_LOG=logs/predictions.jsonl uvicorn src.serving.app:app --port 8000
# interactive docs: http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "applications": [{"SK_ID_CURR": 100001, "AMT_CREDIT": 450000, "EXT_MEAN": 0.25, "NAME_CONTRACT_TYPE": "Cash loans"}],
  "explain_top_k": 3}'
```

Returns a calibrated `default_probability`, the `decision` (approve if below `threshold`), `top_risk_factors` (largest risk-increasing features, in log-odds), `missing_features`, and `warnings` (unknown fields, unseen categories). `GET /health` and `GET /model/info` are for probes and debugging.

**Important:** the API takes the model's *engineered* feature table (the output of `src/features/build_features.py`: application fields plus the aggregates from bureau, previous applications, installments, POS and credit card). A real deployment needs a feature pipeline that builds those aggregates from the applicant's history before calling `/predict`. Fields you leave out are treated as missing, and `missing_features` tells you how many.

## 3. Docker

```bash
docker build -f docker/Dockerfile -t credit-risk-api .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/models" credit-risk-api
```

The model is mounted, not baked in, so a new version is a new folder. Because the model is a joblib pickle, `requirements-serving.txt` pins lightgbm, scikit-learn and pandas; set them to the versions you trained with (print them in Colab).

## 4. Drift monitoring

```bash
python -m src.monitoring.drift_report --current logs/predictions.jsonl
python -m src.monitoring.drift_report --current data/processed/new_batch.parquet --fail-on-alert   # exit 1 on ALERT
```

Computes PSI for the model's 20 most important features (by gain) and for the predicted score, against the reference sample. PSI below 0.10 is stable, 0.10 to 0.25 is a moderate shift, above 0.25 is significant. It also reports missing-rate changes. Output: `reports/drift_report.csv` and `.html`. PSI is unreliable on fewer than about 500 recent rows, and the report says so. Watch `EXT_MEAN` and the `EXT_SOURCE_*` features first, since the model leans on them heavily.

`logs/predictions.jsonl` holds applicant data: decide retention and access rules before enabling it in production.

## Retraining plan

- **Triggers:** an ALERT on the score or on any of the top five features in two consecutive weekly reports; a calendar retrain (for example quarterly); or measured performance decay once outcomes arrive.
- **Labels lag:** defaults take months to show up, so PSI is the early warning and performance can only be checked later on matured cohorts.
- **Process:** rebuild features, run `train.py` as a new MLflow run, compare against the production run, and promote only if test ROC-AUC is not worse than the current model by more than about 0.01 and the reliability curve still looks sound. Keep the previous model folder so rollback is switching `MODEL_DIR`.
- **Re-check the cost assumptions** (`LGD`, `MARGIN`) at each retrain; the threshold depends on them.
