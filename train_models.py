"""
train_models.py
End-to-end ML pipeline for Breast Cancer Wisconsin (Diagnostic) dataset.
Run: python3 train_models.py
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
    classification_report,
)

# ── Step 1: Load data ────────────────────────────────────────────────────────
df = pd.read_csv('breast_cancer.csv')

print("=" * 60)
print("STEP 1: Data Loading & Basic Exploration")
print("=" * 60)
print(f"\nDataset shape : {df.shape}")
print(f"Features      : {df.shape[1] - 1}")
print(f"Instances     : {df.shape[0]}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nClass distribution (diagnosis — 0=malignant, 1=benign):")
print(df['diagnosis'].value_counts())
print(f"\nMissing values:\n{df.isnull().sum().sum()} total")
print(f"\nBasic statistics:")
print(df.describe().round(3))

# ── Step 2: Preprocessing ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preprocessing Pipeline")
print("=" * 60)

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# 80/20 stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler — fit on train only to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train size    : {X_train.shape[0]} samples")
print(f"Test size     : {X_test.shape[0]} samples")
print(f"Train classes : {dict(y_train.value_counts())}")
print(f"Test classes  : {dict(y_test.value_counts())}")
print("Scaler fitted on training data only (no leakage).")

# Save held-out test split (features + diagnosis target) for Streamlit evaluation
test_df = X_test.copy()
test_df['diagnosis'] = y_test.values
test_df.to_csv('test_data.csv', index=False)
print(f"Saved held-out test dataset to test_data.csv ({test_df.shape[0]} rows, {test_df.shape[1]} columns).")

# Persist preprocessing artifacts
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(list(X.columns), 'model/feature_names.joblib')
print("Saved preprocessing artifacts to model/scaler.joblib and model/feature_names.joblib.")

# ── Step 3: Model Training & 5-Fold Cross-Validation ─────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training 5 Classification Models")
print("=" * 60)

classifiers = {
    "Logistic Regression": (
        "logistic_regression.joblib",
        LogisticRegression(C=0.8, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
    ),
    "Decision Tree": (
        "decision_tree.joblib",
        DecisionTreeClassifier(
            criterion='gini', max_depth=5, min_samples_split=6, min_samples_leaf=3, random_state=42
        ),
    ),
    "KNN": (
        "knn.joblib",
        KNeighborsClassifier(n_neighbors=7, weights='distance', metric='minkowski', p=2),
    ),
    "Naive Bayes": (
        "naive_bayes.joblib",
        GaussianNB(var_smoothing=1e-8),
    ),
    "Support Vector Machine": (
        "svm.joblib",
        SVC(C=1.5, kernel='rbf', gamma='scale', probability=True, random_state=42),
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
evaluation_rows = []

for model_name, (filename, clf) in classifiers.items():
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    clf.fit(X_train_scaled, y_train)

    model_path = os.path.join('model', filename)
    joblib.dump(clf, model_path)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    evaluation_rows.append({
        "ML Model Name": model_name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
        "CV Mean Accuracy": round(float(cv_scores.mean()), 4),
        "CV Std": round(float(cv_scores.std()), 4),
    })

    print(f"\n[{model_name}] -> saved to {model_path}")
    print(f"  5-Fold CV Accuracy : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Test Metrics       : Acc={acc:.4f} | AUC={auc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | MCC={mcc:.4f}")
    print(f"  Confusion Matrix   :\n{confusion_matrix(y_test, y_pred)}")

# ── Step 4: Summary Table Export ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Evaluation Metrics Comparison Table")
print("=" * 60)

results_df = pd.DataFrame(evaluation_rows)
results_df.to_csv('model/metrics_summary.csv', index=False)
print(results_df[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]].to_string(index=False))
print("\nSaved metrics summary to model/metrics_summary.csv.")
