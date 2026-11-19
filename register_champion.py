import json
import os

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
STUDENT_USN = "1MS25CI088"

RUN_A_NAME = "A_sentiment_distilbert-sst2"
RUN_B_NAME = "B_sentiment_roberta-large"
SUMMARY_RUN_NAME = "compare_sentiment"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run evaluate.py first.")


def get_latest_run(run_name: str):
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No run found with name '{run_name}'.")
    return runs[0]


run_a = get_latest_run(RUN_A_NAME)
run_b = get_latest_run(RUN_B_NAME)

# Determine winner from outputs/metrics_summary.json if present, else compare_sentiment tag
winner_run_name = RUN_A_NAME
summary_path = os.path.join("outputs", "metrics_summary.json")
if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    winner_run_name = summary_data.get("winner", RUN_A_NAME)
else:
    summary_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{SUMMARY_RUN_NAME}'",
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if summary_runs and summary_runs[0].data.tags.get("winner"):
        winner_run_name = summary_runs[0].data.tags["winner"]


def register_run_model(run, role_label: str):
    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    hf_model_id = run.data.params.get("hf_model_id", "unknown")
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        key="hf_model_id",
        value=str(hf_model_id),
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        key="source_run_name",
        value=role_label,
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        key="student_usn",
        value=STUDENT_USN,
    )
    return mv


mv_a = register_run_model(run_a, RUN_A_NAME)
mv_b = register_run_model(run_b, RUN_B_NAME)

if winner_run_name == RUN_B_NAME:
    champion_mv, challenger_mv = mv_b, mv_a
else:
    champion_mv, challenger_mv = mv_a, mv_b

client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", champion_mv.version)
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", challenger_mv.version)

print(f"Registered '{REGISTERED_MODEL_NAME}' versions: v{mv_a.version} ({RUN_A_NAME}), v{mv_b.version} ({RUN_B_NAME})")
print(f"Alias 'champion'   -> version {champion_mv.version} ({champion_mv.tags.get('hf_model_id')})")
print(f"Alias 'challenger' -> version {challenger_mv.version} ({challenger_mv.tags.get('hf_model_id')})")