import numpy as np
import pandas as pd

from src.monitoring.drift_report import main, psi_categorical, psi_numeric, status
from src.serving.scoring import Scorer


def test_psi_numeric_and_categorical():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 20000))
    b = pd.Series(rng.normal(0, 1, 20000))
    shifted = pd.Series(rng.normal(1.0, 1, 20000))
    assert psi_numeric(a, b) < 0.02
    assert psi_numeric(a, shifted) > 0.25 and status(psi_numeric(a, shifted)) == "ALERT"

    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    assert psi_categorical(ref, ref) < 1e-9
    assert psi_categorical(ref, pd.Series(["a"] * 20 + ["b"] * 80)) > 0.25
    # missing-rate shift is detected too
    with_nan = a.copy()
    with_nan[:8000] = np.nan
    assert psi_numeric(a, with_nan) > 0.25


def test_drift_cli_flags_shifted_data(artifacts, tmp_path):
    model_dir = artifacts["model_dir"]
    ref = pd.read_parquet(model_dir / "reference_sample.parquet")
    assert "score" in ref.columns

    same = ref.sample(1500, random_state=1).drop(columns="score")
    same_path = tmp_path / "same.parquet"
    same.to_parquet(same_path)
    assert main_with(["--model-dir", str(model_dir), "--current", str(same_path),
                      "--out-dir", str(tmp_path / "r1"), "--fail-on-alert"]) == 0

    shifted = same.copy()
    shifted["EXT_MEAN"] = shifted["EXT_MEAN"] - 0.35
    shifted["EXT_SOURCE_2"] = shifted["EXT_SOURCE_2"] - 0.35
    shifted_path = tmp_path / "shifted.parquet"
    shifted.to_parquet(shifted_path)
    assert main_with(["--model-dir", str(model_dir), "--current", str(shifted_path),
                      "--out-dir", str(tmp_path / "r2"), "--fail-on-alert"]) == 1
    rep = pd.read_csv(tmp_path / "r2" / "drift_report.csv")
    assert rep.iloc[0]["status"] == "ALERT"
    assert (tmp_path / "r2" / "drift_report.html").exists()


def main_with(argv):
    import sys
    old = sys.argv
    sys.argv = ["drift_report", *argv]
    try:
        return main()
    finally:
        sys.argv = old
