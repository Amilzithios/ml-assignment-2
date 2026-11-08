# ML Assignment 2 – Breast Cancer Diagnostic Classification & Streamlit Deployment

**Program:** M.Tech (AIML) – Work Integrated Learning Programmes (WILP), BITS Pilani  
**Course:** Machine Learning (Assignment – 2)

---

## a. Problem Statement

Early and accurate differentiation between malignant (cancerous) and benign (non-cancerous) breast masses is critical for clinical oncology decision-making. Manual cytological assessment of fine-needle aspirate (FNA) slides can be subjective and time-consuming. The objective of this project is to build, evaluate, and deploy an end-to-end supervised machine learning classification pipeline that predicts whether a breast tumor is **Malignant (`0`)** or **Benign (`1`)** using 30 quantitative nuclear morphology features extracted from digitized FNA biopsy images. Five distinct classification families—**Logistic Regression**, **Decision Tree**, **K-Nearest Neighbors (KNN)**, **Gaussian Naive Bayes**, and **Support Vector Machine (SVM)**—are trained on a standardized training split, benchmarked across six complementary evaluation metrics on a held-out test set, and served via an interactive **Streamlit** web application without retraining at inference time.

---

## b. Dataset Description

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source Repository:** UCI Machine Learning Repository (accessed via `sklearn.datasets.load_breast_cancer`)
- **Problem Type:** Binary Classification (`diagnosis`: `0` = Malignant, `1` = Benign)
- **Total Instances:** `569` patient biopsy records (Requirement: minimum 500 instances)
  - **Training Split (80%, Stratified):** `455` instances (`285` Benign, `170` Malignant)
  - **Held-Out Test Split (`test_data.csv`, 20%, Stratified):** `114` instances (`72` Benign, `42` Malignant)
- **Feature Size:** `30` continuous numerical features (Requirement: minimum 12 features), with `0` missing values.
- **Feature Groups:** Computed from digitized images of fine-needle aspirates (FNA) of breast masses, describing three statistical aggregations—**Mean** (10 features), **Standard Error** (10 features), and **Worst / Largest** (mean of the three largest values, 10 features)—across ten core nuclear characteristics:
  1. `radius` (mean of distances from center to points on the perimeter)
  2. `texture` (standard deviation of gray-scale values)
  3. `perimeter`
  4. `area`
  5. `smoothness` (local variation in radius lengths)
  6. `compactness` ($\text{perimeter}^2 / \text{area} - 1.0$)
  7. `concavity` (severity of concave portions of the contour)
  8. `concave points` (number of concave portions of the contour)
  9. `symmetry`
  10. `fractal dimension` ("coastline approximation" $- 1$)
- **Preprocessing Applied:** Features are standardized to zero mean and unit variance using `StandardScaler` fitted strictly on the 80% training split (`model/scaler.joblib`) to prevent data leakage before transforming `test_data.csv`.

---

## c. GitHub Repository Link

- **GitHub Repository:** [https://github.com/Amilzithios/ml-assignment-2](https://github.com/Amilzithios/ml-assignment-2)

### Repository Structure

```text
ml-assignment-2/
├── app.py                              # Interactive Streamlit web application
├── train_models.py                     # End-to-end training, CV, evaluation & artifact export script
├── generate_dataset.py                 # Script to export breast_cancer.csv from UCI/sklearn
├── breast_cancer.csv                   # Full dataset (569 rows, 30 features + diagnosis target)
├── test_data.csv                       # Held-out 20% stratified test dataset (114 rows, 31 columns)
├── requirements.txt                    # Python dependencies for Streamlit Community Cloud deployment
├── README.md                           # Project documentation, metrics comparison, and observations
└── model/                              # Saved pre-trained classifiers, source code & preprocessing artifacts
    ├── train_models.py                 # Model training & evaluation source script
    ├── scaler.joblib                   # Fitted StandardScaler (trained on X_train only)
    ├── feature_names.joblib            # Ordered list of the 30 input feature names
    ├── logistic_regression.joblib      # Trained Logistic Regression model
    ├── decision_tree.joblib            # Trained Decision Tree Classifier
    ├── knn.joblib                      # Trained K-Nearest Neighbors Classifier
    ├── naive_bayes.joblib              # Trained Gaussian Naive Bayes Classifier
    ├── svm.joblib                      # Trained Support Vector Machine (RBF) Classifier
    └── metrics_summary.csv             # Exported evaluation metrics summary table
```

---

## d. Models Used: Evaluation Metrics Comparison Table

All five classification models were trained on the same stratified training set (`455` rows) and evaluated on the held-out test set (`test_data.csv`, `114` rows):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9957** | **0.9861** | 0.9861 | **0.9861** | **0.9623** |
| **Decision Tree** | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| **KNN** | 0.9737 | 0.9884 | 0.9600 | **1.0000** | 0.9796 | 0.9442 |
| **Naive Bayes** | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| **Support Vector Machine** | **0.9825** | 0.9954 | **0.9861** | 0.9861 | **0.9861** | **0.9623** |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Achieved tied-highest test Accuracy (`0.9825`), F1 (`0.9861`), and MCC (`0.9623`), along with the highest overall ROC-AUC (`0.9957`) and 5-fold CV accuracy (`0.9758 ± 0.0128`). L2 regularization (`C=0.8`) on standardized features effectively handles multicollinearity among correlated geometric features (radius, perimeter, area), producing only 2 errors (1 FP, 1 FN) out of 114 test samples. |
| **Decision Tree** | Recorded the lowest test Accuracy (`0.9035`), AUC (`0.9373`), and MCC (`0.7969`), with 11 misclassifications (4 FP, 7 FN). Because axis-aligned splits partition one feature at a time, a single tree struggles to capture smooth diagonal decision boundaries across 30 correlated continuous features even when regularized (`max_depth=5`, `min_samples_leaf=3`). |
| **KNN** | Delivered strong performance with `0.9737` Accuracy, `0.9884` AUC, `0.9442` MCC, and a perfect `1.0000` Recall on the Benign class (72/72 identified, 3 FP, 0 FN). Distance weighting (`k=7`, Euclidean metric) benefits substantially from `StandardScaler` normalization, though local neighborhood voting remains slightly sensitive to boundary points in 30-dimensional space. |
| **Naive Bayes** | Attained `0.9298` Accuracy, `0.9444` F1, and `0.8492` MCC (4 FP, 4 FN), while still maintaining a high ranking capability (`0.9868` AUC). Its primary limitation on this dataset is the strong conditional independence assumption, which is violated by highly collinear nuclear features (`mean/worst radius`, `perimeter`, `area`, and `concavity`). |
| **Support Vector Machine** | Matched Logistic Regression across all thresholded metrics—Accuracy (`0.9825`), Precision (`0.9861`), Recall (`0.9861`), F1 (`0.9861`), and MCC (`0.9623`)—with an AUC of `0.9954` (41/42 Malignant and 71/72 Benign correctly classified). The RBF kernel (`C=1.5`) constructs a robust maximum-margin boundary in standardized feature space with excellent generalization (`5-Fold CV = 0.9692`). |
| **Overall, Winner for your dataset?** | **Logistic Regression** (closely tied with **Support Vector Machine**) is the overall winner for this dataset. Both achieve the top Accuracy (`98.25%`), F1 (`0.9861`), and MCC (`0.9623`), while **Logistic Regression** edges ahead with the highest ROC-AUC (`0.9957`), highest 5-fold cross-validation stability (`0.9758`), lower computational complexity, and direct clinical interpretability via calibrated feature coefficients. |

---

## Live Streamlit App Link

- **Live Deployed Application:** [https://amilzith-ml-assignment-2.streamlit.app](https://amilzith-ml-assignment-2.streamlit.app)

### Local Execution Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-run training pipeline to regenerate models and test_data.csv
python3 train_models.py

# 3. Launch the interactive Streamlit web application
streamlit run app.py
```
