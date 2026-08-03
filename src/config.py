import os

MLFLOW_TRACKING_URI = "sqlite:///" + os.path.join(os.path.dirname(__file__), "..", "mlflow.db")
EXPERIMENT_NAME = "churn-retraining"
REGISTERED_MODEL_NAME = "churn-classifier"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
NEW_DATA_PATH = os.path.join(DATA_DIR, "new_batch.csv")