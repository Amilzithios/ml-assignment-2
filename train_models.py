"""
train_models.py
End-to-end ML pipeline for Breast Cancer Wisconsin (Diagnostic) dataset.
Run: python3 train_models.py
"""

import pandas as pd
import numpy as np

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
