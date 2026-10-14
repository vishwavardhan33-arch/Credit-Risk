"""Model loading and scoring, kept free of any web framework so it is easy to test and reuse."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.calibration import apply_calibration

ID_FIELD = "SK_ID_CURR"


def clean(v: Any):
    """Make a value JSON-safe (NaN -> None, numpy scalars -> Python)."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (int, bool, str)):
        return v
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


class Scorer:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self.config = json.loads((d / "model_config.json").read_text())
        self.model = joblib.load(d / "lgbm_model.joblib")
        self.columns: list[str] = self.config["feature_columns"]
        self.levels: dict[str, list[str]] = self.config["categorical_levels"]
        self.threshold: float = float(self.config["threshold"])
        self.calibration: dict = self.config["calibration"]
        self.version: str = self.config.get("model_version", "unknown")

    # -- input handling -------------------------------------------------------
    def align(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with exactly the training columns, order and dtypes."""
        out = pd.DataFrame(index=df.index)
        for c in self.columns:
            s = df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
            if c in self.levels:
                s = s.astype(object)
                s = s.where(s.isin(self.levels[c]), None)      # unseen or missing -> missing
                out[c] = pd.Categorical(s, categories=self.levels[c])
            else:
                out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        return out

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Calibrated default probability for each row of a feature table."""
        raw = self.model.predict_proba(self.align(df))[:, 1]
        return apply_calibration(raw, self.calibration)

    # -- request scoring ------------------------------------------------------
    def predict(self, records: list[dict], explain_top_k: int = 3):
        """Score records. Returns (predictions, aligned_frame)."""
        df = pd.DataFrame(records)          # keeps empty records as rows (from_records drops them)
        X = self.align(df)
        raw = self.model.predict_proba(X)[:, 1]
        p = apply_calibration(raw, self.calibration)

        contrib = None
        if explain_top_k > 0:
            # per-feature contributions in log-odds (last column is the base value)
            contrib = self.model.booster_.predict(X, pred_contrib=True)[:, :-1]

        known = set(self.columns) | {ID_FIELD}
        preds = []
        for i, rec in enumerate(records):
            warnings = []
            unknown = sorted(set(rec) - known)
            if unknown:
                warnings.append(f"ignored unknown fields: {unknown}")
            for c in self.levels:
                if c in df.columns and pd.notna(df.at[df.index[i], c]) and pd.isna(X.at[X.index[i], c]):
                    warnings.append(f"unseen category in {c}: {df.at[df.index[i], c]!r} treated as missing")
            item = {
                "id": clean(rec.get(ID_FIELD)),
                "default_probability": round(float(p[i]), 6),
                "decision": "approve" if p[i] < self.threshold else "reject",
                "threshold": self.threshold,
                "missing_features": int(X.iloc[i].isna().sum()),
                "warnings": warnings,
            }
            if contrib is not None:
                order = np.argsort(-contrib[i])[:explain_top_k]
                item["top_risk_factors"] = [
                    {"feature": self.columns[j], "contribution": round(float(contrib[i, j]), 4),
                     "value": clean(X.iloc[i, j])}
                    for j in order if contrib[i, j] > 0
                ]
            preds.append(item)
        return preds, X
