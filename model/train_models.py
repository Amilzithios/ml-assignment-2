"""
model/train_models.py
Model training, 5-fold cross-validation, evaluation, and artifact serialization script
for the Breast Cancer Wisconsin (Diagnostic) dataset.
Can be executed from project root (`python3 model/train_models.py`) or inside `model/`.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "breast_cancer.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

test_df = X_test.copy()
test_df["diagnosis"] = y_test.values
test_df.to_csv(TEST_DATA_PATH, index=False)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.joblib"))

classifiers = {
    "Logistic Regression": (
        "logistic_regression.joblib",
        LogisticRegression(C=0.8, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42),
    ),
    "Decision Tree": (
        "decision_tree.joblib",
        DecisionTreeClassifier(
            criterion="gini", max_depth=5, min_samples_split=6, min_samples_leaf=3, random_state=42
        ),
    ),
    "KNN": (
        "knn.joblib",
        KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski", p=2),
    ),
    "Naive Bayes": (
        "naive_bayes.joblib",
        GaussianNB(var_smoothing=1e-8),
    ),
    "Support Vector Machine": (
        "svm.joblib",
        SVC(C=1.5, kernel="rbf", gamma="scale", probability=True, random_state=42),
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for model_name, (filename, clf) in classifiers.items():
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="accuracy")
    clf.fit(X_train_scaled, y_train)
    joblib.dump(clf, os.path.join(MODEL_DIR, filename))

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    rows.append({
        "ML Model Name": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
        "CV Mean Accuracy": round(float(cv_scores.mean()), 4),
        "CV Std": round(float(cv_scores.std()), 4),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(MODEL_DIR, "metrics_summary.csv"), index=False)
print(results_df[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].to_string(index=False))

