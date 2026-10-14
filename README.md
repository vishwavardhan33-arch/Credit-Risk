# Credit Risk: Model Selection Playbook

Predicting loan default on Home Credit data, comparing model families on accuracy, speed, latency, interpretability and maintenance cost.

## Results table
| Model | ROC-AUC | PR-AUC | Fit time (s) | Inference latency | Interpretability |
|-------|---------|--------|--------------|-------------------|------------------|
|       |         |        |              |                   |                  |

## Decision guide
- Small tabular data, need interpretability ->
- Large tabular data, max accuracy ->
- Text ->

## Phases
1. EDA, baselines, model ladder
2. Multi-table feature engineering
3. Calibration, cost-based threshold, SHAP
4. FastAPI, Docker, MLflow, drift monitoring
