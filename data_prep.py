"""Loads the credit card fraud dataset and prepares a train/test split.

Shared by every modeling notebook so they all use the exact same cleaning,
split, and scaling - a fair, apples-to-apples comparison across models.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "creditcard.csv"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop_duplicates()


def prepare_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Returns X_train, X_test, y_train, y_test.

    Two things matter here that wouldn't for a balanced dataset:

    1. `stratify=y`: a plain random split could easily end up with, say, 60
       fraud cases in training and only 10 in testing (or vice versa) purely
       by chance, since fraud is only 0.17% of the data. Stratifying forces
       both the train and test sets to keep that same ~0.17% fraud rate, so
       the test set is a fair, representative sample.
    2. Scaling `Time` and `Amount`: the V1-V28 columns are already roughly
       standardized (that's a side effect of how PCA was computed), but
       `Time` (up to ~172,000) and `Amount` (up to ~25,000) are on wildly
       different scales. Left unscaled, they can dominate a model like
       logistic regression simply by having bigger raw numbers, not because
       they're more informative - and the underlying optimizer converges
       poorly on such mismatched scales.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    return X_train, X_test, y_train, y_test
