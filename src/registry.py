"""
Model registry logic: registering new versions, deciding whether a candidate
should be promoted to production, and rolling back if needed.
"""
import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "churn-classifier"
ALIAS_PRODUCTION = "production"
ALIAS_STAGING = "staging"
ALIAS_PREVIOUS_PRODUCTION = "previous-production"

MIN_ACCEPTABLE_F1 = 0.70
PROMOTION_MIN_IMPROVEMENT = 0.005  # candidate must beat prod by at least 0.5% F1


def get_client() -> MlflowClient:
    return MlflowClient()


def register_candidate(run_id: str, model_name: str = REGISTERED_MODEL_NAME):
    """Registers the model artifact from a run as a new (unpromoted) version."""
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = get_client()
    client.set_registered_model_alias(model_name, ALIAS_STAGING, result.version)
    return result.version


def get_current_production_metrics(client: MlflowClient, model_name: str = REGISTERED_MODEL_NAME):
    """Returns (version, metrics) for whatever is aliased 'production', or (None, None)."""
    try:
        mv = client.get_model_version_by_alias(model_name, ALIAS_PRODUCTION)
    except Exception:
        return None, None
    run = client.get_run(mv.run_id)
    return mv.version, run.data.metrics


def promote_if_better(candidate_version: str, candidate_metrics: dict,
                       model_name: str = REGISTERED_MODEL_NAME, metric_key: str = "f1") -> dict:
    """The promotion gate. Returns a dict describing the decision made."""
    client = get_client()
    candidate_score = candidate_metrics[metric_key]

    decision = {
        "candidate_version": candidate_version,
        "candidate_score": candidate_score,
        "promoted": False,
        "reason": None,
    }

    if candidate_score < MIN_ACCEPTABLE_F1:
        decision["reason"] = f"Below absolute floor of {MIN_ACCEPTABLE_F1}. Not promoted."
        return decision

    prod_version, prod_metrics = get_current_production_metrics(client, model_name)

    if prod_version is None:
        client.set_registered_model_alias(model_name, ALIAS_PRODUCTION, candidate_version)
        decision["promoted"] = True
        decision["reason"] = "No existing production model; promoted as the first version."
        return decision

    prod_score = prod_metrics[metric_key]
    improvement = candidate_score - prod_score
    decision["previous_production_version"] = prod_version
    decision["improvement"] = improvement

    if improvement >= PROMOTION_MIN_IMPROVEMENT:
        client.set_registered_model_alias(model_name, ALIAS_PREVIOUS_PRODUCTION, prod_version)
        client.set_registered_model_alias(model_name, ALIAS_PRODUCTION, candidate_version)
        decision["promoted"] = True
        decision["reason"] = f"Improved F1 by {improvement:.4f}. Promoted; v{prod_version} kept for rollback."
    else:
        decision["reason"] = f"Improvement of {improvement:.4f} below required {PROMOTION_MIN_IMPROVEMENT}. Unchanged."

    return decision


def rollback(model_name: str = REGISTERED_MODEL_NAME) -> dict:
    """Reverts 'production' back to whatever is tagged 'previous-production'."""
    client = get_client()
    try:
        prev = client.get_model_version_by_alias(model_name, ALIAS_PREVIOUS_PRODUCTION)
    except Exception:
        return {"rolled_back": False, "reason": "No previous-production version recorded."}
    current = client.get_model_version_by_alias(model_name, ALIAS_PRODUCTION)
    client.set_registered_model_alias(model_name, ALIAS_PRODUCTION, prev.version)
    return {"rolled_back": True, "restored_version": prev.version, "demoted_version": current.version}