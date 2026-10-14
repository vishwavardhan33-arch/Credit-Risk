"""Drift monitoring with the Population Stability Index (PSI), the standard metric in credit scoring.

Compares recent applications against the reference sample saved by training, for the model's most
important features and for the predicted default probability.

    python -m src.monitoring.drift_report --current logs/predictions.jsonl
    python -m src.monitoring.drift_report --current data/processed/new_batch.parquet --fail-on-alert

Rule of thumb: PSI < 0.10 stable, 0.10 to 0.25 moderate shift, > 0.25 significant shift.
PSI is noisy on small samples (below roughly 500 rows), so the report says so.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.serving.scoring import Scorer

EPS = 1e-4
WARN, ALERT = 0.10, 0.25
MIN_ROWS = 500


def _psi_from_shares(ref_share: np.ndarray, cur_share: np.ndarray) -> float:
    r, c = np.clip(ref_share, EPS, None), np.clip(cur_share, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.astype(object).where(ref.notna(), "<missing>").astype(str).value_counts(normalize=True)
    c = cur.astype(object).where(cur.notna(), "<missing>").astype(str).value_counts(normalize=True)
    cats = r.index.union(c.index)
    return _psi_from_shares(r.reindex(cats, fill_value=0).to_numpy(), c.reindex(cats, fill_value=0).to_numpy())


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = pd.to_numeric(ref, errors="coerce"), pd.to_numeric(cur, errors="coerce")
    if ref.dropna().nunique() <= 10:          # flags and small discrete codes: compare values directly
        return psi_categorical(ref, cur)
    edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(s: pd.Series) -> np.ndarray:
        counts = np.histogram(s.dropna(), bins=edges)[0].astype(float)
        counts = np.append(counts, s.isna().sum())          # missing values get their own bucket
        return counts / max(1, counts.sum())

    return _psi_from_shares(shares(ref), shares(cur))


def status(psi: float) -> str:
    return "ALERT" if psi > ALERT else "WARN" if psi > WARN else "ok"


def load_current(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]   # prediction log
    df = pd.DataFrame([r["features"] for r in rows])
    df["score"] = [r["score"] for r in rows]
    return df


def top_features(scorer: Scorer, n: int) -> list[str]:
    gain = scorer.model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(-gain)[:n]
    return [scorer.columns[i] for i in order]


def build_report(scorer: Scorer, reference: pd.DataFrame, current: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for rank, c in enumerate(top_features(scorer, top_n), start=1):
        if c not in reference.columns or c not in current.columns:
            continue
        is_cat = c in scorer.levels
        value = psi_categorical(reference[c], current[c]) if is_cat else psi_numeric(reference[c], current[c])
        rows.append({"feature": c, "importance_rank": rank, "psi": round(value, 4), "status": status(value),
                     "ref_missing_pct": round(100 * reference[c].isna().mean(), 1),
                     "cur_missing_pct": round(100 * current[c].isna().mean(), 1)})

    if "score" in reference.columns:
        cur_score = current["score"] if "score" in current.columns else pd.Series(scorer.score_frame(current))
        value = psi_numeric(reference["score"], cur_score)
        rows.append({"feature": "score (calibrated default probability)", "importance_rank": 0,
                     "psi": round(value, 4), "status": status(value),
                     "ref_missing_pct": 0.0, "cur_missing_pct": 0.0})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def to_html(report: pd.DataFrame, notes: list[str]) -> str:
    colors = {"ALERT": "#f8d7da", "WARN": "#fff3cd", "ok": "#ffffff"}
    head = "".join(f"<th>{html.escape(c)}</th>" for c in report.columns)
    body = "".join(
        f"<tr style='background:{colors[r.status]}'>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
        for r in report.itertuples(index=False))
    note_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return (f"<html><body style='font-family:sans-serif'><h2>Drift report (PSI)</h2><ul>{note_html}</ul>"
            f"<table border='1' cellpadding='4' style='border-collapse:collapse'><tr>{head}</tr>{body}</table>"
            f"<p>PSI &lt; {WARN} stable, {WARN}-{ALERT} moderate, &gt; {ALERT} significant.</p></body></html>")


def main() -> int:
    ap = argparse.ArgumentParser(description="PSI drift report for the credit risk model.")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--reference", default=None, help="default: <model-dir>/reference_sample.parquet")
    ap.add_argument("--current", required=True, help=".jsonl prediction log, .parquet or .csv of feature rows")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--fail-on-alert", action="store_true", help="exit with code 1 if any PSI > 0.25")
    a = ap.parse_args()

    scorer = Scorer(a.model_dir)
    reference = pd.read_parquet(a.reference or Path(a.model_dir) / "reference_sample.parquet")
    current = load_current(a.current)
    notes = [f"reference rows: {len(reference):,}", f"current rows: {len(current):,}",
             f"model version: {scorer.version}"]
    if len(current) < MIN_ROWS:
        notes.append(f"WARNING: only {len(current)} current rows, PSI is unreliable below ~{MIN_ROWS}")
    if "score" not in reference.columns:
        notes.append("reference has no 'score' column, so score drift was skipped")
    elif "score" in current.columns or len(current):
        notes.append(f"mean predicted default: reference {reference['score'].mean():.4f}"
                     + (f", current {current['score'].mean():.4f}" if "score" in current.columns else ""))

    report = build_report(scorer, reference, current, a.top_n)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "drift_report.csv", index=False)
    (out / "drift_report.html").write_text(to_html(report, notes))

    print("\n".join(notes))
    print(report.to_string(index=False))
    n_alert = int((report["status"] == "ALERT").sum())
    print(f"\n{n_alert} feature(s) in ALERT, {int((report['status'] == 'WARN').sum())} in WARN. Saved to {out}/")
    return 1 if (a.fail_on_alert and n_alert) else 0


if __name__ == "__main__":
    raise SystemExit(main())
