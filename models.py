"""Model training and evaluation, shared across the modeling notebooks.

Pulling this out of the notebooks means every notebook trains and scores
a model the exact same way - no risk of a copy-pasted metric call silently
drifting between notebooks 02-05.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def train_logistic_regression(X_train, y_train, class_weight=None) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
    model.fit(X_train, y_train)
    return model


def apply_smote(X_train, y_train, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def train_random_forest(X_train, y_train, n_estimators: int = 200) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
