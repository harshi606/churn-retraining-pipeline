"""
Integration tests for the promotion gate — exercises the real MLflow
registry logic against a temporary, disposable tracking store (not your
real mlflow.db), so each test starts from a clean slate.
"""
import numpy as np
import pytest
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

from src.registry import (
    REGISTERED_MODEL_NAME,
    ALIAS_PRODUCTION,
    get_client,
    register_candidate,
    promote_if_better,
    rollback,
)


@pytest.fixture
def mlflow_tracking(tmp_path):
    """Points MLflow at a throwaway SQLite store that only this test sees."""
    db_path = tmp_path / "test_mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("test-experiment")
    yield


def _log_dummy_run(metrics: dict) -> str:
    """
    Logs a trivially trained model but with metrics we control directly.
    We don't care if the model is actually good — we're testing the GATE's
    decision logic, not model quality, so we just need a valid registerable
    artifact with metrics we choose.
    """
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, size=10)
    model = LogisticRegression().fit(X, y)

    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        return run.info.run_id


def test_first_model_is_always_promoted(mlflow_tracking):
    run_id = _log_dummy_run({"f1": 0.75})
    version = register_candidate(run_id)
    decision = promote_if_better(version, {"f1": 0.75})

    assert decision["promoted"] is True
    assert "No existing production model" in decision["reason"]


def test_weak_candidate_is_rejected(mlflow_tracking):
    v1 = register_candidate(_log_dummy_run({"f1": 0.80}))
    promote_if_better(v1, {"f1": 0.80})

    # Barely ties -- should NOT clear the required improvement margin
    v2 = register_candidate(_log_dummy_run({"f1": 0.801}))
    decision = promote_if_better(v2, {"f1": 0.801})

    assert decision["promoted"] is False

    client = get_client()
    prod = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_PRODUCTION)
    assert prod.version == v1  # production unchanged


def test_below_floor_candidate_is_rejected_even_as_first_model(mlflow_tracking):
    version = register_candidate(_log_dummy_run({"f1": 0.5}))
    decision = promote_if_better(version, {"f1": 0.5})

    assert decision["promoted"] is False
    assert "Below absolute floor" in decision["reason"]


def test_better_candidate_promotes_and_rollback_restores_previous(mlflow_tracking):
    v1 = register_candidate(_log_dummy_run({"f1": 0.80}))
    promote_if_better(v1, {"f1": 0.80})

    v2 = register_candidate(_log_dummy_run({"f1": 0.90}))
    decision = promote_if_better(v2, {"f1": 0.90})
    assert decision["promoted"] is True

    client = get_client()
    prod = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_PRODUCTION)
    assert prod.version == v2

    rb = rollback()
    assert rb["rolled_back"] is True
    assert rb["restored_version"] == v1

    prod_after = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_PRODUCTION)
    assert prod_after.version == v1