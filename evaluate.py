import json
import os
import random
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from transformers import pipeline

# 0. Reproducibility & Directories
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

STUDENT_USN = "1MS25CI088"
DATASET_NAME = "eval_inputs.csv"
PRIMARY_METRIC = "f1_macro"
MARGIN = 0.01  # 1 percentage point minimum improvement required over Champion

# 1. Setup MLflow Tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "hf-sentiment-champion-challenger"
mlflow.set_experiment(experiment_name)

# 2. Load Evaluation Data
print("Loading evaluation data...")
eval_path = os.path.join("data", DATASET_NAME)
eval_df = pd.read_csv(eval_path)
eval_df["label_norm"] = eval_df["label"].astype(str).str.strip().str.upper()

device_idx = 0 if torch.cuda.is_available() else -1
device_name = "cuda" if torch.cuda.is_available() else "cpu"

library_versions_tag = (
    f"transformers=={transformers.__version__}, "
    f"torch=={torch.__version__}, "
    f"mlflow=={mlflow.__version__}"
)

# 3. Define Models
models_config = {
    "A_sentiment_distilbert-sst2": {
        "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion",
    },
    "B_sentiment_roberta-large": {
        "hf_model_id": "siebert/sentiment-roberta-large-english",
        "role": "challenger",
    },
}

results_summary = {}
predictions_by_run = {}

# 4. Run evaluation for each model
for run_name, config in models_config.items():
    print(f"\n--- Running model: {run_name} ({config['hf_model_id']}) ---")

    with mlflow.start_run(run_name=run_name):
        # Log Parameters
        mlflow.log_param("hf_model_id", config["hf_model_id"])
        mlflow.log_param("task", "sentiment-classification")
        mlflow.log_param("role", config["role"])
        mlflow.log_param("eval_samples", len(eval_df))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("device", device_name)
        mlflow.log_param("truncation", True)

        # Log Tags
        mlflow.set_tag("student_usn", STUDENT_USN)
        mlflow.set_tag("library_versions", library_versions_tag)
        mlflow.set_tag("dataset_name", DATASET_NAME)

        # Load Hugging Face pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model=config["hf_model_id"],
            device=device_idx,
            truncation=True,
        )

        predictions = []
        scores = []
        latencies = []

        for text in eval_df["text"]:
            start_time = time.perf_counter()
            result = classifier(str(text))[0]
            end_time = time.perf_counter()

            raw_label = str(result["label"]).upper()
            if "LABEL_1" in raw_label or raw_label == "POS":
                label = "POSITIVE"
            elif "LABEL_0" in raw_label or raw_label == "NEG":
                label = "NEGATIVE"
            else:
                label = raw_label

            predictions.append(label)
            scores.append(float(result.get("score", 0.0)))
            latencies.append((end_time - start_time) * 1000.0)

        run_preds_df = eval_df[["text", "label"]].copy()
        run_preds_df["prediction"] = predictions
        run_preds_df["score"] = scores
        run_preds_df["correct"] = run_preds_df["prediction"] == eval_df["label_norm"]
        predictions_by_run[run_name] = run_preds_df

        acc = float(accuracy_score(eval_df["label_norm"], predictions))
        f1_mac = float(f1_score(eval_df["label_norm"], predictions, average="macro", zero_division=0))
        avg_latency = float(np.mean(latencies))
        total_inference = float(np.sum(latencies) / 1000.0)

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_mac)
        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_metric("total_inference_s", total_inference)

        # Save & log predictions CSV
        pred_path = os.path.join("outputs", f"predictions_{run_name}.csv")
        run_preds_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path, artifact_path="outputs")

        # Save & log Confusion Matrix plot
        cm = confusion_matrix(
            eval_df["label_norm"],
            predictions,
            labels=["NEGATIVE", "POSITIVE"],
        )
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NEGATIVE", "POSITIVE"])
        disp.plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title(f"Confusion Matrix: {run_name}")
        cm_path = os.path.join("outputs", f"confusion_matrix_{run_name}.png")
        fig_cm.tight_layout()
        fig_cm.savefig(cm_path)
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="outputs")

        # Log shared eval_inputs.csv and README if present
        mlflow.log_artifact(eval_path, artifact_path="data")
        if os.path.exists("README.md"):
            mlflow.log_artifact("README.md")

        # Log model
        mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="model",
        )

        results_summary[run_name] = {
            "hf_model_id": config["hf_model_id"],
            "role": config["role"],
            "accuracy": acc,
            "f1_macro": f1_mac,
            "avg_latency_ms": avg_latency,
            "total_inference_s": total_inference,
        }

        print(
            f"Finished {run_name} | Accuracy: {acc:.4f} | "
            f"Macro-F1: {f1_mac:.4f} | Avg Latency: {avg_latency:.2f}ms"
        )

# 5. Summary / Comparison Run (compare_sentiment)
champ_key = "A_sentiment_distilbert-sst2"
chall_key = "B_sentiment_roberta-large"

champ_metrics = results_summary[champ_key]
chall_metrics = results_summary[chall_key]

delta_primary = chall_metrics[PRIMARY_METRIC] - champ_metrics[PRIMARY_METRIC]
# Challenger wins if it beats Champion primary metric by at least MARGIN;
# tie-breaker: if within margin, prefer faster model (lower avg_latency_ms)
if delta_primary >= MARGIN:
    winner = chall_key
elif abs(delta_primary) < MARGIN and chall_metrics["avg_latency_ms"] < champ_metrics["avg_latency_ms"]:
    winner = chall_key
else:
    winner = champ_key

# Save metrics_summary.json
metrics_summary_path = os.path.join("outputs", "metrics_summary.json")
with open(metrics_summary_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "primary_metric": PRIMARY_METRIC,
            "margin": MARGIN,
            "delta_primary_metric": delta_primary,
            "winner": winner,
            "models": results_summary,
        },
        f,
        indent=2,
    )

# Save comparison bar chart (Plot #2)
fig_bar, axes = plt.subplots(1, 2, figsize=(10, 4))
short_names = ["Model A (DistilBERT)", "Model B (RoBERTa-large)"]

axes[0].bar(short_names, [champ_metrics["f1_macro"], chall_metrics["f1_macro"]])
axes[0].set_title("Macro-F1 (Higher is Better)")
axes[0].set_ylim(0, 1.05)

axes[1].bar(short_names, [champ_metrics["avg_latency_ms"], chall_metrics["avg_latency_ms"]])
axes[1].set_title("Avg Latency ms (Lower is Better)")

bar_path = os.path.join("outputs", "comparison_metrics.png")
fig_bar.tight_layout()
fig_bar.savefig(bar_path)
plt.close(fig_bar)

# Save qualitative disagreements (up to 5 cases where A and B disagree)
preds_a = predictions_by_run[champ_key]["prediction"]
preds_b = predictions_by_run[chall_key]["prediction"]
disagree_df = pd.DataFrame(
    {
        "text": eval_df["text"],
        "reference": eval_df["label_norm"],
        "model_a_pred": preds_a,
        "model_b_pred": preds_b,
        "comment": "",
    }
)
disagree_df = disagree_df[disagree_df["model_a_pred"] != disagree_df["model_b_pred"]].head(5)
disagree_path = os.path.join("outputs", "disagreements_sample.csv")
disagree_df.to_csv(disagree_path, index=False)

with mlflow.start_run(run_name="compare_sentiment"):
    mlflow.log_param("primary_metric", PRIMARY_METRIC)
    mlflow.log_param("margin", MARGIN)
    mlflow.log_metric("delta_primary_metric", delta_primary)
    mlflow.log_metric("champion_f1_macro", champ_metrics["f1_macro"])
    mlflow.log_metric("challenger_f1_macro", chall_metrics["f1_macro"])
    mlflow.log_metric("delta_avg_latency_ms", chall_metrics["avg_latency_ms"] - champ_metrics["avg_latency_ms"])

    mlflow.set_tag("student_usn", STUDENT_USN)
    mlflow.set_tag("library_versions", library_versions_tag)
    mlflow.set_tag("dataset_name", DATASET_NAME)
    mlflow.set_tag("winner", winner)

    mlflow.log_artifact(metrics_summary_path, artifact_path="outputs")
    mlflow.log_artifact(bar_path, artifact_path="outputs")
    mlflow.log_artifact(disagree_path, artifact_path="outputs")

print("\nEvaluation complete! Champion, Challenger, and compare_sentiment logged to MLflow.")
print(f"Winner ({PRIMARY_METRIC}, margin={MARGIN}): {winner} (delta={delta_primary:+.4f})")