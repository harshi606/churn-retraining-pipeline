"""
Loads and splits data for training/evaluation.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "churned"
RANDOM_SEED = 42


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def train_val_split(df: pd.DataFrame, test_size: float = 0.2):
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )


def combine_datasets(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Used when retraining on original data + newly arrived data."""
    return pd.concat(dfs, ignore_index=True)