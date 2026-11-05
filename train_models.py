"""
train_models.py
End-to-end ML pipeline for Breast Cancer Wisconsin (Diagnostic) dataset.
Run: python3 train_models.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
