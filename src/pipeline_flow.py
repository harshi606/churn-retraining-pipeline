"""
The orchestrated retraining pipeline: load data -> train -> register -> promote.

Run standalone:
    python -m src.pipeline_flow
"""
import os
import mlflow
from prefect import flow, task, get_run_logger

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, NEW_DATA_PATH, TRAIN_DATA_PATH
from src.data_ingestion import combine_datasets, load_csv, train_val_split
from src.registry import promote_if_better, register_candidate
from src.train import train_and_log


@task(retries=1, retry_delay_seconds=10)
def load_data_task():
    logger = get_run_logger()
    train_df = load_csv(TRAIN_DATA_PATH)
    logger.info(f"Loaded base training data: {len(train_df)} rows")

    if os.path.exists(NEW_DATA_PATH):
        new_df = load_csv(NEW_DATA_PATH)
        logger.info(f"Loaded new batch data: {len(new_df)} rows")
        return combine_datasets(train_df, new_df)
    return train_df


@task
def split_task(df):
    return train_val_split(df)


@task
def train_task(X_train, y_train, X_val, y_val, run_name: str):
    logger = get_run_logger()
    run_id, metrics, _ = train_and_log(X_train, y_train, X_val, y_val, run_name=run_name)
    logger.info(f"Trained candidate run {run_id} with metrics: {metrics}")
    return run_id, metrics


@task
def register_task(run_id: str):
    return register_candidate(run_id)


@task
def promotion_gate_task(version, metrics):
    logger = get_run_logger()
    decision = promote_if_better(version, metrics)
    logger.info(f"Promotion decision: {decision}")
    return decision


@flow(name="churn-model-retraining")
def retraining_pipeline(run_name: str = "automated-retrain"):
    logger = get_run_logger()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data_task()
    X_train, X_val, y_train, y_val = split_task(df)
    run_id, metrics = train_task(X_train, y_train, X_val, y_val, run_name)
    version = register_task(run_id)
    decision = promotion_gate_task(version, metrics)

    logger.info(f"Pipeline complete. v{version} {'PROMOTED' if decision['promoted'] else 'NOT promoted'}.")
    return decision


if __name__ == "__main__":
    print(retraining_pipeline())