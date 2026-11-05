"""
app.py
Interactive Streamlit Web Application for Breast Cancer Wisconsin (Diagnostic) Classification
M.Tech (AIML) - Machine Learning Assignment 2 | BITS Pilani WILP
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="OncoMetric | Breast Cancer ML Diagnostic Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom UI Styling ────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
        color: #f8fafc;
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
    }
    .hero-banner p {
        margin: 0.4rem 0 0 0;
        font-size: 0.98rem;
        color: #cbd5e1;
    }
    .metric-tile {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 0.6rem;
    }
    .metric-tile .label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        font-weight: 600;
    }
    .metric-tile .value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.15rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #dbeafe;
        color: #1e40af;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model Registry & Artifacts Loader ────────────────────────────────────────
MODEL_REGISTRY = {
    "Logistic Regression": {
        "file": "model/logistic_regression.joblib",
        "family": "Linear Discriminative Classifier (L2 Regularized, C=0.8)",
    },
    "Decision Tree": {
        "file": "model/decision_tree.joblib",
        "family": "Non-Parametric CART Tree (Gini, max_depth=5, min_samples_leaf=3)",
    },
    "KNN": {
        "file": "model/knn.joblib",
        "family": "Distance-Weighted K-Nearest Neighbors (k=7, Euclidean)",
    },
    "Naive Bayes": {
        "file": "model/naive_bayes.joblib",
        "family": "Gaussian Probabilistic Classifier (var_smoothing=1e-8)",
    },
    "Support Vector Machine": {
        "file": "model/svm.joblib",
        "family": "RBF Kernel Margin Classifier (C=1.5, gamma='scale')",
    },
}


@st.cache_resource
def load_saved_artifacts():
    """Load pre-fitted scaler, feature names, and all 5 pre-trained classifiers."""
    scaler = joblib.load("model/scaler.joblib")
    feature_names = joblib.load("model/feature_names.joblib")
    loaded_models = {}
    for name, meta in MODEL_REGISTRY.items():
        loaded_models[name] = joblib.load(meta["file"])
    return scaler, feature_names, loaded_models


def compute_model_metrics(clf, X_scaled, y_true):
    """Evaluate a pre-trained model without retraining."""
    y_pred = clf.predict(X_scaled)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_scaled)[:, 1]
    else:
        y_prob = clf.decision_function(X_scaled)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return metrics, y_pred, y_prob


# ── Header Banner ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <h1>🧬 OncoMetric: Breast Cancer Diagnostic Evaluation Suite</h1>
        <p>BITS Pilani WILP · M.Tech (AIML) Machine Learning Assignment 2 · Pre-Trained Multi-Model Evaluation (No Retraining on Test Upload)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    scaler, expected_features, models = load_saved_artifacts()
except Exception as err:
    st.error(
        f"Failed to load saved model artifacts from `model/`. Ensure `python3 train_models.py` has been run. Details: {err}"
    )
    st.stop()

# ── Sidebar Controls (Upload + Model Selection) ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Evaluation Controls")

    st.subheader("1. Upload Test Dataset (CSV)")
    uploaded_csv = st.file_uploader(
        "Upload test CSV containing 30 features + `diagnosis` column",
        type=["csv"],
        help="Upload test_data.csv (or custom test split with the same 30 features and 'diagnosis' target).",
    )

    if os.path.exists("test_data.csv"):
        with open("test_data.csv", "rb") as f_test:
            st.download_button(
                label="⬇️ Download Sample test_data.csv",
                data=f_test,
                file_name="test_data.csv",
                mime="text/csv",
                width="stretch",
            )

    st.divider()
    st.subheader("2. Select Classification Model")
    selected_model_name = st.selectbox(
        "Choose a trained classifier to inspect:",
        options=list(MODEL_REGISTRY.keys()),
        index=0,
    )
    st.caption(f"**Architecture:** {MODEL_REGISTRY[selected_model_name]['family']}")

    st.divider()
    st.markdown(
        """
        **Target Encoding (`diagnosis`):**
        - `0` → **Malignant** (Cancerous)
        - `1` → **Benign** (Non-cancerous)
        """
    )

# ── Load & Validate Test Data ────────────────────────────────────────────────
if uploaded_csv is not None:
    eval_df = pd.read_csv(uploaded_csv)
    data_source_label = f"Uploaded CSV (`{uploaded_csv.name}`)"
elif os.path.exists("test_data.csv"):
    eval_df = pd.read_csv("test_data.csv")
    data_source_label = "Bundled Held-Out Test Split (`test_data.csv`)"
else:
    st.warning("Please upload `test_data.csv` in the sidebar to begin evaluation.")
    st.stop()

missing_cols = [c for c in expected_features if c not in eval_df.columns]
if missing_cols or "diagnosis" not in eval_df.columns:
    st.error(
        "The uploaded CSV is missing required columns. "
        f"Missing features: {missing_cols[:5]} | Target `'diagnosis'` present: {'diagnosis' in eval_df.columns}"
    )
    st.stop()

X_eval = eval_df[expected_features]
y_eval = eval_df["diagnosis"].astype(int)

# Transform using pre-fitted training scaler (strictly no refitting)
X_eval_scaled = scaler.transform(X_eval)

# Status Bar
benign_count = int((y_eval == 1).sum())
malignant_count = int((y_eval == 0).sum())
st.markdown(
    f'<span class="status-pill">📂 Data Source: {data_source_label}</span>'
    f'<span class="status-pill">🧪 Test Samples: {len(eval_df)}</span>'
    f'<span class="status-pill">🔢 Features: {len(expected_features)}</span>'
    f'<span class="status-pill">🩺 Class Balance: {benign_count} Benign (1) / {malignant_count} Malignant (0)</span>',
    unsafe_allow_html=True,
)

with st.expander("🔍 Preview Active Test Dataset (First 8 Rows)", expanded=False):
    st.dataframe(eval_df.head(8), width="stretch")

# ── Selected Model Evaluation Section ────────────────────────────────────────
st.markdown("---")
st.subheader(f"📊 Detailed Evaluation: {selected_model_name}")
st.caption(
    f"Loaded artifact: `{MODEL_REGISTRY[selected_model_name]['file']}` · "
    f"Specification: {MODEL_REGISTRY[selected_model_name]['family']}"
)

active_clf = models[selected_model_name]
active_metrics, y_pred_active, y_prob_active = compute_model_metrics(
    active_clf, X_eval_scaled, y_eval
)

metric_cols = st.columns(6)
metric_labels = [
    ("Accuracy", active_metrics["Accuracy"]),
    ("AUC Score", active_metrics["AUC"]),
    ("Precision", active_metrics["Precision"]),
    ("Recall", active_metrics["Recall"]),
    ("F1 Score", active_metrics["F1"]),
    ("MCC Score", active_metrics["MCC"]),
]

for col, (m_name, m_val) in zip(metric_cols, metric_labels):
    with col:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="label">{m_name}</div>
                <div class="value">{m_val:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Confusion Matrix & Classification Report Side-by-Side ────────────────────
viz_col1, viz_col2 = st.columns([1.05, 1.15])

with viz_col1:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_eval, y_pred_active)
    cm_pct = cm / cm.sum() * 100.0
    annot_labels = np.array(
        [
            [
                f"TN: {cm[0, 0]}\n({cm_pct[0, 0]:.1f}%)",
                f"FP: {cm[0, 1]}\n({cm_pct[0, 1]:.1f}%)",
            ],
            [
                f"FN: {cm[1, 0]}\n({cm_pct[1, 0]:.1f}%)",
                f"TP: {cm[1, 1]}\n({cm_pct[1, 1]:.1f}%)",
            ],
        ]
    )

    fig_cm, ax_cm = plt.subplots(figsize=(5.2, 4.0))
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted Diagnosis", fontsize=10, fontweight="bold")
    ax_cm.set_ylabel("Actual Diagnosis", fontsize=10, fontweight="bold")
    ax_cm.set_title(f"{selected_model_name} — Confusion Matrix", fontsize=11, pad=10)
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with viz_col2:
    st.markdown("#### Classification Report")
    report_dict = classification_report(
        y_eval,
        y_pred_active,
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    st.dataframe(
        report_df.style.format(
            {
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}",
            }
        ),
        width="stretch",
        height=235,
    )

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    st.info(
        f"**Clinical Diagnostic Summary ({selected_model_name}):** "
        f"Correctly identified **{tp}/{tp + fn}** Benign cases (Recall = `{active_metrics['Recall']:.4f}`) and "
        f"**{tn}/{tn + fp}** Malignant cases (Specificity = `{specificity:.4f}`), with only **{fp + fn}** total misclassifications out of {len(y_eval)} test records."
    )

# ── Cross-Model Comparison on Test Data ──────────────────────────────────────
st.markdown("---")
st.subheader("🏆 All 5 Classification Models — Test Data Benchmark")
st.caption(
    "Side-by-side evaluation of all five saved models on the active test dataset without retraining."
)

all_rows = []
roc_curves_data = {}

for m_name, clf_obj in models.items():
    m_scores, _, m_prob = compute_model_metrics(clf_obj, X_eval_scaled, y_eval)
    all_rows.append(
        {
            "ML Model Name": m_name,
            "Accuracy": round(m_scores["Accuracy"], 4),
            "AUC": round(m_scores["AUC"], 4),
            "Precision": round(m_scores["Precision"], 4),
            "Recall": round(m_scores["Recall"], 4),
            "F1": round(m_scores["F1"], 4),
            "MCC": round(m_scores["MCC"], 4),
        }
    )
    fpr, tpr, _ = roc_curve(y_eval, m_prob)
    roc_curves_data[m_name] = (fpr, tpr, m_scores["AUC"])

comparison_df = pd.DataFrame(all_rows)
metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.dataframe(
    comparison_df.style.highlight_max(
        subset=metric_columns, color="#dcfce7"
    ).format({col: "{:.4f}" for col in metric_columns}),
    width="stretch",
    hide_index=True,
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("#### Multi-Metric Performance Comparison")
    melted_df = comparison_df.melt(
        id_vars="ML Model Name", var_name="Metric", value_name="Score"
    )
    fig_bar, ax_bar = plt.subplots(figsize=(6.4, 4.0))
    sns.barplot(
        data=melted_df,
        x="Metric",
        y="Score",
        hue="ML Model Name",
        palette="viridis",
        ax=ax_bar,
    )
    ax_bar.set_ylim(0.75, 1.02)
    ax_bar.set_ylabel("Metric Score")
    ax_bar.set_xlabel("")
    ax_bar.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

with chart_col2:
    st.markdown("#### ROC Curves Across All 5 Classifiers")
    fig_roc, ax_roc = plt.subplots(figsize=(6.4, 4.0))
    for m_name, (fpr, tpr, auc_val) in roc_curves_data.items():
        lw = 2.6 if m_name == selected_model_name else 1.5
        ax_roc.plot(fpr, tpr, linewidth=lw, label=f"{m_name} (AUC = {auc_val:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Baseline (0.5000)")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig_roc)
    plt.close(fig_roc)
