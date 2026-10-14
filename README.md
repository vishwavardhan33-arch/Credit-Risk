# Credit Default Risk: Model Selection Playbook, End to End

Predicting loan default on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) data, built to answer a practical question: **which model should you use, and what else does a working ML engineer have to get right around it?**

The project compares model families on more than accuracy, engineers features from six relational tables, calibrates the probabilities, sets the approval threshold from a cost model instead of 0.5, explains the model with SHAP, and then serves it with FastAPI, tracks runs in MLflow and monitors drift with PSI.

## Results at a glance

| | |
|---|---|
| Final model | LightGBM, 175 features, sigmoid-calibrated, `CODE_GENDER` removed |
| Held-out test ROC-AUC / PR-AUC | **0.782 / 0.281** (PR-AUC is about 3.5x the 8.07% default rate) |
| Approval threshold | 0.136 predicted default probability (cost-based) |
| At that threshold, on the test set | approves 83.3% of applicants; 4.8% of approved applicants default vs 8.1% overall; rejects 50.8% of all defaulters |
| Profit vs approving everyone | +22.3%, **under assumed** loss and margin figures (see [Limitations](#limitations)) |
| Tests | 11 pytest tests (features, training, API, drift) |

## Dataset

307,511 loan applications, 8.07% defaulted. The raw files are **not** in this repo: accept the competition rules on Kaggle and download them yourself. Six tables are used (`application_train`, `bureau`, `previous_application`, `installments_payments`, `POS_CASH_balance`, `credit_card_balance`); `bureau_balance` is not used.

All model selection uses 5-fold stratified cross-validation on an 80% training split. The 20% test split (61,503 rows) is touched only once, after the calibrator, threshold and feature set were fixed.

## What I found

### 1. Model ladder (application table only)

| Model | ROC-AUC | PR-AUC | Fit time per fold | Interpretability |
|---|---|---|---|---|
| Dummy (prior) | 0.5000 | 0.0807 | 0.1 s | n/a |
| Logistic regression | 0.7456 | 0.2202 | 12.4 s | coefficients |
| Decision tree (depth 6) | 0.7128 | 0.1855 | 11.4 s | readable rules |
| Random forest (200 trees) | 0.7374 | 0.2192 | 82.6 s | needs SHAP |
| LightGBM | **0.7569** | **0.2402** | 28.7 s | needs SHAP |

(Fit times are from a Colab CPU runtime. Inference latency and model size were not measured.)

- **PR-AUC is the headline metric.** With 8% positives, ROC-AUC looks flattering; the dummy model's PR-AUC equals the base rate, so LightGBM's 0.24 is about 3x lift.
- **A linear model is a serious contender on one table.** Logistic regression trails LightGBM by 0.011 ROC-AUC and is fully interpretable.
- **The random forest was the worst value:** slowest by far, and no better than logistic regression.

### 2. Better data beat a better model

Same LightGBM settings, same CV, only the features change:

| Feature set | ROC-AUC | PR-AUC | Fit time per fold |
|---|---|---|---|
| Application table only | 0.7569 | 0.2402 | 28.7 s |
| + domain ratios, bureau, previous applications, installments | 0.7642 | 0.2477 | 49.7 s |
| + POS cash and credit card balances (all 176 features) | **0.7788** | **0.2692** | 60.4 s |

Every history table is aggregated to one row per applicant before joining, and no feature uses the target, which prevents leakage. These are sequential stages, not a controlled ablation of each table; the optional ablation cell in `notebooks/02_features.ipynb` runs that.

### 3. Calibration

Plain LightGBM trained without class reweighting came out already well calibrated: mean predicted default 8.00% vs 8.07% observed, Brier score 0.06652 raw vs 0.06653 after sigmoid calibration, which is a tie. The pipeline still compares sigmoid and isotonic calibration on out-of-fold scores and stores the winner, but the effect here is close to a no-op. It would matter if you trained with `scale_pos_weight` or `is_unbalance`.

### 4. Threshold from a cost model, not 0.5

Approving a borrower who defaults costs `LGD x loan amount`; rejecting one who would have repaid costs `MARGIN x loan amount`. With `LGD = 0.60` and `MARGIN = 0.10`, the break-even probability is `MARGIN / (MARGIN + LGD) = 0.143`, and the empirical profit-maximizing threshold on out-of-fold predictions is 0.136. Across plausible assumptions (LGD 0.4 to 0.8, margin 0.05 to 0.15) the best threshold moves between roughly 0.05 and 0.27, so these two inputs matter and should come from the lender.

### 5. Explainability (SHAP)

- `EXT_MEAN` (the mean of the three external scores) dominates: mean |SHAP| of 0.43 against 0.099 for the next feature. The model leans heavily on third-party scores, so they are the first thing to watch for drift.
- Several engineered history features reach the top 15: `INST_LATE_FLAG_MEAN`, `PREV_APP_CREDIT_RATIO_MEAN`, `PREV_AMT_ANNUITY_MEAN`, `BUREAU_DEBT_CREDIT_RATIO`. No POS or credit card feature appears there, so their +0.015 ROC-AUC gain is not attributed to specific features.

### 6. Fairness check: `CODE_GENDER`

On the held-out set the first model was well calibrated within each group, yet approval rates differed (women 86.5% vs men 76.9%; under-30s about 70% vs over-60s 94.7%), and gender was the fourth most important feature. I retrained without it:

| | With gender | Without gender |
|---|---|---|
| Out-of-fold ROC-AUC | 0.7790 | 0.7776 |
| Test ROC-AUC | 0.7820 | 0.7817 |
| Test PR-AUC | 0.2801 | 0.2805 |
| Defaults rejected | 51.1% | 50.8% |
| Profit vs approve-all | +22.5% | +22.3% |

The cost is under 0.002 ROC-AUC, so the shipped model drops it. Removing a column does not guarantee equal outcomes, because other features can act as proxies. The group-level approval gaps were **not** re-measured on the no-gender model, and age is still an input.

### 7. Serving, tracking, monitoring

- **API** (`src/serving`): `POST /predict` takes one or more applicants and returns a calibrated default probability, an approve/reject decision, the top risk-increasing features (log-odds contributions), a count of missing features and warnings for unknown fields or unseen categories.
- **Experiment tracking** (`src/models/train.py`): each run logs parameters, out-of-fold and test metrics, the threshold and the model files to MLflow, so variants such as `--drop CODE_GENDER` can be compared side by side.
- **Drift monitoring** (`src/monitoring/drift_report.py`): PSI on the 20 most important features and on the score, against a reference sample with out-of-fold scores. Rule of thumb: below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 significant. The CLI can exit non-zero on an alert for scheduled jobs.

Drift demo, replaying 800 held-out applicants: normal traffic gave 0 alerts (largest PSI 0.085) and a mean predicted default of 8.3%. After lowering the external scores by 0.30, five alerts fired (`EXT_MEAN` PSI 5.19), mean predicted default doubled to 16.7%, and the rejection rate rose from 15% to 47%. This is a simulated shift, not production traffic.

<!-- Add screenshots here once saved:
![Drift report with alerts](reports/figures/drift_alert.png)
![MLflow run comparison](reports/figures/mlflow_compare.png)
-->

## Choosing a model: what this project supports

- **Tabular data with good feature engineering:** gradient boosting (LightGBM) gave the best accuracy here, at moderate cost.
- **Interpretability or regulation matters most:** logistic regression was within 0.011 ROC-AUC on the application table, and its coefficients are directly explainable. A boosted model needs SHAP.
- **A random forest was not the right middle ground** on this data: slowest to fit, no better than the linear model.
- **Spend effort on data before tuning.** Adding history tables gained 0.022 ROC-AUC; model choice among the better candidates gained about 0.01.
- Text and time-series model choices are outside this project's evidence and are not covered.

## Repository layout

```
credit-risk/
├── notebooks/          01_eda_baselines, 02_features, 03_calibration_threshold_shap (executed, Colab)
├── src/
│   ├── features/       build_features.py  (feature table from the six CSVs)
│   ├── models/         calibration.py, train.py  (training, threshold, MLflow)
│   ├── serving/        scoring.py, app.py  (FastAPI)
│   └── monitoring/     drift_report.py  (PSI)
├── scripts/            send_sample.py  (replays held-out applicants against the API)
├── tests/              11 pytest tests on synthetic data
├── docker/             Dockerfile
├── docs/PHASE4.md      serving, monitoring and retraining runbook
├── requirements.txt, requirements-serving.txt
└── data/ models/ logs/ reports/   (git-ignored)
```

## Reproduce it

Commands are for macOS/Linux; on Windows PowerShell activate with `.venv\Scripts\Activate.ps1` and set variables with `$env:MODEL_DIR="models"`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. features: put the six Kaggle CSVs in data/raw/, then
python -m src.features.build_features          # writes data/processed/train_features.parquet

# 2. tests (synthetic data, about 10 seconds)
python -m pytest tests -q

# 3. train, calibrate, set threshold, log to MLflow
python -m src.models.train --features data/processed/train_features.parquet --model-dir models_nogender \
       --drop CODE_GENDER --run-name no-gender
mlflow ui --backend-store-uri sqlite:///mlflow.db

# 4. serve and score
mkdir -p logs
MODEL_DIR=models_nogender PREDICTION_LOG=logs/predictions.jsonl uvicorn src.serving.app:app --port 8000
MODEL_DIR=models_nogender python scripts/send_sample.py 5          # in a second terminal

# 5. drift report on the logged requests
python -m src.monitoring.drift_report --model-dir models_nogender --current logs/predictions.jsonl
```

Docker (the model folder is mounted, not baked in). `requirements-serving.txt` pins lightgbm, scikit-learn and pandas to the versions used for training, because the model is a joblib pickle; set them to match yours before building:

```bash
docker build -f docker/Dockerfile -t credit-risk-api .
docker run --rm -p 8000:8000 -v "$(pwd)/models_nogender:/models" credit-risk-api
```

More detail on the API, monitoring thresholds and the retraining plan is in [`docs/PHASE4.md`](docs/PHASE4.md).

## Limitations

- **Cost assumptions are placeholders.** `LGD = 0.60` and `MARGIN = 0.10` are illustrative; the profit figures and the 0.136 threshold depend on them.
- **Random validation split.** The dataset has no usable application dates, so the split is stratified random. A real deployment needs out-of-time validation.
- **No hyperparameter tuning.** The LightGBM settings are reasonable defaults. Note that `subsample` has no effect without `subsample_freq`, so row bagging is inactive.
- **Heavy reliance on external scores** (`EXT_SOURCE_*`), which makes the model only as stable as those inputs.
- **The API expects engineered features**, not raw application data. A real deployment needs a pipeline that builds the history aggregates per applicant before calling it. Fields left out are treated as missing and counted in `missing_features`.
- **Fairness work is partial:** gender was removed, but proxies and age were not audited after the change.
- **The prediction log contains applicant data;** decide retention and access rules before using it outside a demo.
- This is a learning and portfolio project on a public dataset, not a production credit decision system.

---<img width="440" height="408" alt="Screenshot 2026-10-14 114145" src="https://github.com/user-attachments/assets/8c3ac05c-ffb3-485d-91e9-af50e9c0936a" />


Built by Doni Vishwa Vardhan. Data: Home Credit Group via Kaggle, used under the competition rules.
