# Automated Churn Model Retraining Pipeline

An end-to-end MLOps pipeline that retrains a customer churn model, evaluates
it against the model currently in production, and only promotes it if it's
genuinely better — with rollback support if a promoted model needs to be reverted.

## Why this exists

Training a model once and shipping it is easy. The harder, more realistic
problem is: what happens when new data arrives? This pipeline automates that
decision safely — a retrained model only replaces production if it clears
an absolute quality floor *and* beats the current production model by a
meaningful margin, not just a coin-flip tie.

## Architecture
New data arrives → combine with base training data → train candidate model
→ log to MLflow (params, metrics, model artifact)
→ register as a new model version
→ promotion gate:
- below quality floor? → reject
- beats current production? → promote, keep old version for rollback
- doesn't beat production? → reject, production unchanged
**Stack:**
- **Prefect** — orchestrates the pipeline as a sequence of tracked, retryable tasks
- **MLflow** — experiment tracking (every training run is permanently logged)
  and model registry (explicit `production` / `staging` / `previous-production`
  aliases determine what's actually live)
- **scikit-learn** — the model itself (Random Forest classifier) is intentionally
  simple; the pipeline around it is the point

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python data/generate_data.py
```

## Running the pipeline

```bash
python -m src.pipeline_flow
```

This loads the base training data plus any new batch data, trains a candidate
model, registers it, and runs it through the promotion gate.

## The promotion gate

A candidate model is promoted only if:
1. It clears an absolute minimum F1 score (0.70), **and**
2. It beats the current production model's F1 by at least 0.5% absolute —
   not just ties with it (this prevents production from flapping on noise).

Before promotion, the current production version is preserved under a
`previous-production` alias, enabling rollback.

## Rollback

```python
from src.registry import rollback
rollback()  # reverts production to the previously promoted version
```

## Tests

```bash
pytest tests/ -v
```

Integration tests exercise the real promotion gate logic (promote, reject,
rollback) against a temporary, disposable MLflow store.

## Viewing experiment history

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Possible extensions

- Swap local SQLite/file storage for a hosted MLflow server + S3 artifact store
- Add scheduled execution via a Prefect deployment (`prefect deployment build`)
- Add a data drift check as a gate *before* retraining is even triggered
- Canary rollout: serve the new model to a % of traffic before full promotion