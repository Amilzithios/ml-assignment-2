"""
Generate breast_cancer.csv from sklearn's built-in Breast Cancer Wisconsin dataset.
Source: UCI ML Repository (via sklearn)
Features: 30 | Instances: 569 | Task: Binary Classification
Run once: python3 generate_dataset.py
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target  # 0 = malignant, 1 = benign

df.to_csv('breast_cancer.csv', index=False)

print(f"Dataset saved: breast_cancer.csv")
print(f"Shape        : {df.shape}")
print(f"Class counts :\n{df['diagnosis'].value_counts().rename({0: 'malignant', 1: 'benign'})}")
print(f"Features     : {list(data.feature_names)}")
