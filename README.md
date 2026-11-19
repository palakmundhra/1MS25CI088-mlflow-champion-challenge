# Champion–Challenger Sentiment Classification with Hugging Face and MLflow

## Task and models
- **Task**: Sentiment classification (binary: `POSITIVE` / `NEGATIVE`)
- **Dataset & Subset Size**: Shared evaluation set in `data/eval_inputs.csv` (10 samples) and unseen inference set in `data/new_data.csv` (8 samples)
- **Model A (`A_sentiment_distilbert-sst2`)**: `distilbert-base-uncased-finetuned-sst-2-english` (~67M parameters, champion candidate)
- **Model B (`B_sentiment_roberta-large`)**: `siebert/sentiment-roberta-large-english` (~355M parameters, challenger candidate)

## How to run
Run these commands in order inside the project folder (`1MS25CI088-mlflow-champion-challenger`):

1. Start the MLflow server:
   ```bash
   python -m mlflow server --host 127.0.0.1 --port 5000


## Champion rule
Written before the final run:
- **Primary metric**: `f1_macro` (higher is better)[cite: 1].
- **Minimum improvement margin**: Challenger (`B_sentiment_roberta-large`) must beat Champion (`A_sentiment_distilbert-sst2`) on `f1_macro` by at least `0.01` so small random noise does not trigger a switch[cite: 1].
- **Tie-breaker / latency guard**: If the `f1_macro` difference is within `0.01`, prefer the faster model with lower `avg_latency_ms` (lower is better)[cite: 1].
- **Metric directions**: Higher is better for `f1_macro` and `accuracy`; lower is better for `avg_latency_ms` and `total_inference_s`[cite: 1].

## Results table

| Metric / Criterion | Model A (`distilbert-sst2`) | Model B (`roberta-large`) | Difference (B − A) | Direction |
| :--- | :---: | :---: | :---: | :---: |
| `f1_macro` | 0.8990 | 1.0000 | +0.1010 | Higher is better |
| `accuracy` | 0.9000 | 1.0000 | +0.1000 | Higher is better |
| `avg_latency_ms` | 75.33 ms | 415.23 ms | +339.90 ms | Lower is better |
| `total_inference_s` | 0.7533 s | 4.1523 s | +3.3990 s | Lower is better |

## Decision
Model B (`siebert/sentiment-roberta-large-english`, registered as version 2) wins and becomes `@champion`, while Model A (`distilbert-base-uncased-finetuned-sst-2-english`, version 1) becomes `@challenger`[cite: 1, 2, 3]. Model B reaches `1.0000` on both `f1_macro` and `accuracy`, compared to Model A's `0.8990` `f1_macro` and `0.9000` `accuracy`. That is a `+0.1010` gain on `f1_macro`, which passes the `0.01` threshold[cite: 1].

Because the evaluation set has only 10 samples, a single error changes accuracy by `0.1000`, so sample size is a limitation even though the improvement is clear here[cite: 1]. The trade-off is speed and size: Model B (~355M parameters) takes `415.23 ms` per sample on average (`4.15 s` total), about `5.5x` slower than Model A (~67M parameters) at `75.33 ms` per sample (`0.75 s` total), plus longer model-loading time[cite: 1]. Because accuracy is prioritized here over speed, Model B is promoted[cite: 1].

## Limitations and declaration
- **Limitations**: The evaluation set is small (10 rows) and ran on CPU, making RoBERTa-large noticeably slower and heavier to load[cite: 1]. Also, `data/new_data.csv` has no ground-truth labels and binary sentiment can treat mild/neutral phrasing as positive (for example, *"The presentation was decent enough, nothing groundbreaking though."* scored `POSITIVE` at `0.9844`)[cite: 1, 3].
- **Declaration**: Completed for the MLOps with MLflow mini-project (`1MS25CI088`)[cite: 1]. An AI assistant was used to help write, structure, and review code, documentation, and steps; everything was tested and verified, and I can explain each part in the viva[cite: 1].