"""
Trains a churn classifier and logs it to MLflow as a run.

Note: this only creates a run — it does NOT register or promote the model.
That's a deliberate separation of concerns, handled later in registry.py.
Training produces a candidate; promotion is a separate decision.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn

RANDOM_SEED = 42


def train_model(X_train, y_train, n_estimators: int = 200, max_depth: int = 8):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_SEED,
        class_weight="balanced",  # churn is imbalanced (~25%)
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val) -> dict:
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    return {
        "f1": f1_score(y_val, preds),
        "precision": precision_score(y_val, preds),
        "recall": recall_score(y_val, preds),
        "roc_auc": roc_auc_score(y_val, probs),
    }


def train_and_log(X_train, y_train, X_val, y_val, run_name: str,
                   n_estimators: int = 200, max_depth: int = 8):
    """
    Trains, evaluates, and logs params/metrics/model as one MLflow run.
    Returns (run_id, metrics_dict, model).
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})

        model = train_model(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)
        metrics = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.iloc[:5])

        return run.info.run_id, metrics, model