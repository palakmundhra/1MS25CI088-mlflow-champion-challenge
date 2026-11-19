import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "sentiment-model"
ALIAS = "champion"
NEW_DATA_PATH = os.path.join("data", "new_data.csv")
OUTPUT_PATH = os.path.join("outputs", "new_data_predictions.csv")

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# 1. Inspect alias metadata (version + HF model ID) without hardcoding version/run IDs
mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)
hf_model_id = mv.tags.get("hf_model_id")
if not hf_model_id and mv.run_id:
    run = client.get_run(mv.run_id)
    hf_model_id = run.data.params.get("hf_model_id", "unknown")

print(f"Loaded alias '@{ALIAS}' for '{REGISTERED_MODEL_NAME}'")
print(f"Model Version : {mv.version}")
print(f"HF Model ID   : {hf_model_id or 'unknown'}")

# 2. Load model via MLflow alias URI (no direct Hugging Face calls here)
model_uri = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# 3. Load unseen data and run predictions
new_df = pd.read_csv(NEW_DATA_PATH)
raw_preds = loaded_model.predict(new_df["text"].tolist())

# Normalize output format cleanly into a DataFrame
if isinstance(raw_preds, pd.DataFrame):
    preds_df = raw_preds.copy()
    if "text" not in preds_df.columns:
        preds_df.insert(0, "text", new_df["text"])
elif isinstance(raw_preds, list) and len(raw_preds) > 0 and isinstance(raw_preds[0], dict):
    labels = []
    scores = []
    for item in raw_preds:
        lbl = str(item.get("label", "")).upper()
        if "LABEL_1" in lbl or lbl == "POS":
            lbl = "POSITIVE"
        elif "LABEL_0" in lbl or lbl == "NEG":
            lbl = "NEGATIVE"
        labels.append(lbl)
        if "score" in item:
            scores.append(float(item["score"]))
    preds_df = pd.DataFrame({"text": new_df["text"], "prediction": labels})
    if scores:
        preds_df["score"] = scores
else:
    labels = []
    for p in raw_preds:
        lbl = str(p).upper()
        if "LABEL_1" in lbl or lbl == "POS":
            lbl = "POSITIVE"
        elif "LABEL_0" in lbl or lbl == "NEG":
            lbl = "NEGATIVE"
        labels.append(lbl)
    preds_df = pd.DataFrame({"text": new_df["text"], "prediction": labels})

preds_df["model_version"] = mv.version
preds_df["hf_model_id"] = hf_model_id or "unknown"

os.makedirs("outputs", exist_ok=True)
preds_df.to_csv(OUTPUT_PATH, index=False)

print("\n--- Predictions on New Data ---")
print(preds_df.to_string(index=False))
print(f"\nSaved predictions to {OUTPUT_PATH}")