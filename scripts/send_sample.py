import json
import os
import sys
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
shift = "--shift" in sys.argv
model_dir = os.environ.get("MODEL_DIR", "models")
url = os.environ.get("API_URL", "http://localhost:8000") + "/predict"

cfg = json.load(open(f"{model_dir}/model_config.json"))
df = pd.read_parquet("data/processed/train_features.parquet")
_, test_idx = train_test_split(df.index, test_size=0.2, stratify=df["TARGET"], random_state=42)
rows = df.loc[test_idx].sample(n, random_state=0)
if shift:
    for c in ["EXT_MEAN", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        rows[c] = rows[c] - 0.30
records = json.loads(rows[cfg["feature_columns"] + ["SK_ID_CURR"]].to_json(orient="records"))
outcomes = rows["TARGET"].tolist()

results = []
for i in range(0, n, 200):
    body = json.dumps({"applications": records[i:i + 200], "explain_top_k": 3 if n <= 10 else 0}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    results += json.load(urllib.request.urlopen(req))["predictions"]

if n <= 10:
    for p, actual in zip(results, outcomes):
        top = p["top_risk_factors"][0]["feature"] if p["top_risk_factors"] else "-"
        print(f"id {p['id']}: PD {p['default_probability']:.1%} -> {p['decision']:7s} | actually defaulted: {bool(actual)} | top risk factor: {top}")
else:
    approved = sum(p["decision"] == "approve" for p in results)
    print(f"sent {n} applicants: {approved} approved, {n - approved} rejected")