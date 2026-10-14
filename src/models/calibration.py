"""Calibration and cost-based threshold helpers, shared by training and serving.

Calibrators are stored as plain parameters (no pickled classes), so the API can
apply them with numpy alone.
"""
from __future__ import annotations

import numpy as np


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrator(kind: str, scores, y) -> dict:
    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    if kind == "sigmoid":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(scores).reshape(-1, 1), y)
        return {"kind": "sigmoid", "coef": float(m.coef_[0, 0]), "intercept": float(m.intercept_[0])}
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, y)
        return {"kind": "isotonic", "x": m.X_thresholds_.tolist(), "y": m.y_thresholds_.tolist()}
    raise ValueError(f"unknown calibrator kind: {kind}")


def apply_calibration(p, params: dict) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if params["kind"] == "sigmoid":
        return 1 / (1 + np.exp(-(params["coef"] * logit(p) + params["intercept"])))
    return np.interp(p, params["x"], params["y"])


def cv_calibrated(kind: str, scores, y, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Calibrate each fold with a calibrator fitted on the other folds (honest comparison)."""
    from sklearn.model_selection import StratifiedKFold
    scores, y = np.asarray(scores, dtype=float), np.asarray(y)
    out = np.zeros_like(scores)
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(scores, y):
        out[va] = apply_calibration(scores[va], fit_calibrator(kind, scores[tr], y[tr]))
    return out


def choose_calibrator(scores, y, kinds=("sigmoid", "isotonic")):
    """Pick the calibrator with the lowest cross-validated Brier score.

    Returns (best_kind, params_fitted_on_all_scores, {kind: cv_calibrated_scores}).
    """
    from sklearn.metrics import brier_score_loss
    cv_scores = {k: cv_calibrated(k, scores, y) for k in kinds}
    best = min(cv_scores, key=lambda k: brier_score_loss(y, cv_scores[k]))
    return best, fit_calibrator(best, scores, y), cv_scores


def total_profit(p, y, credit, thr: float, lgd: float, margin: float) -> float:
    """Profit if we approve everyone with p < thr.

    A repaid loan earns margin * amount, a defaulted loan loses lgd * amount.
    """
    p, y, credit = np.asarray(p), np.asarray(y), np.asarray(credit, dtype=float)
    per_loan = np.where(y == 0, margin * credit, -lgd * credit)
    return float(per_loan[p < thr].sum())


def choose_threshold(p, y, credit, lgd: float, margin: float, grid=None) -> dict:
    grid = np.linspace(0.01, 0.60, 118) if grid is None else np.asarray(grid)
    profits = np.array([total_profit(p, y, credit, t, lgd, margin) for t in grid])
    return {
        "threshold": float(grid[profits.argmax()]),
        "analytic": margin / (margin + lgd),
        "grid": grid,
        "profits": profits,
    }
