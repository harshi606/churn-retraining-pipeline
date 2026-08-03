"""
Generates a synthetic "customer churn" dataset.

Two files are produced:
  - train.csv: the initial training set
  - new_batch.csv: a later batch of data with mild drift, simulating
    a month passing and customer behavior shifting slightly. This is
    what justifies having an automated retraining pipeline at all.
"""
import os
import pandas as pd
from sklearn.datasets import make_classification

FEATURE_NAMES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "contract_length_score",
    "avg_session_minutes",
    "num_products",
    "late_payments_last_year",
    "satisfaction_score",
    "competitor_offer_seen",
]


def _make_dataset(n_samples: int, seed: int, drift_shift: float = 0.0) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURE_NAMES),
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.75, 0.25],  # churn (minority class) ~25%, like real churn data
        flip_y=0.03,
        class_sep=1.1,
        random_state=seed,
    )
    X = X + drift_shift  # simulate drift in later data
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["churned"] = y
    return df


def generate(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    train_df = _make_dataset(n_samples=4000, seed=42, drift_shift=0.0)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    new_df = _make_dataset(n_samples=1200, seed=99, drift_shift=0.35)
    new_df.to_csv(os.path.join(output_dir, "new_batch.csv"), index=False)

    print(f"train.csv: {len(train_df)} rows, churn rate {train_df['churned'].mean():.3f}")
    print(f"new_batch.csv: {len(new_df)} rows, churn rate {new_df['churned'].mean():.3f}")


if __name__ == "__main__":
    generate(os.path.dirname(__file__))