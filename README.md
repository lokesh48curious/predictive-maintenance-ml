# Predictive Maintenance — Machine Failure Prediction

ML pipeline on the **AI4I 2020 Predictive Maintenance Dataset** that predicts
machine failure from sensor readings, with a Streamlit app for real-time
predictions.

## Pipeline

1. Load and inspect the AI4I dataset (shape, dtypes, missing values, class balance)
2. Clean the data (drop duplicates, drop ID/leakage columns, impute missing values)
3. One-hot encode categorical features (`Type`: L/M/H)
4. Train/test split (stratified, 80/20)
5. SMOTE applied **only** to the training set (test set stays untouched/realistic)
6. Train Logistic Regression, Random Forest, and XGBoost
7. Evaluate: confusion matrices, precision, recall, F1, ROC-AUC, ROC curves
8. Feature importance: native (RF, XGBoost) + SHAP summary plot
9. Save the best model (by ROC-AUC) + scaler + metadata
10. Streamlit interface for real-time prediction

## Setup

```bash
pip install -r requirements.txt
```

Download `ai4i2020.csv` from the
[UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
and place it at `data/ai4i2020.csv`.

## Run

```bash
# 1. Train models, generate plots, save best model
python train_models.py

# 2. Launch the prediction app
streamlit run app.py
```

## Outputs

- `outputs/confusion_matrices.png` — confusion matrix per model
- `outputs/roc_curves.png` — ROC curves overlaid, all 3 models
- `outputs/feature_importance.png` — RF + XGBoost native importances
- `outputs/shap_summary.png` — SHAP summary plot (XGBoost)
- `outputs/metrics.json` — precision/recall/F1/ROC-AUC per model
- `models/best_model.joblib` — best model by ROC-AUC
- `models/scaler.joblib` — StandardScaler fit on training data
- `models/model_meta.json` — which model won, feature list, its metrics

## Notes on design choices

- **Target leakage**: `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are dropped because
  they're sub-labels of *which* failure mode occurred — near-perfect
  predictors of `Machine failure` that wouldn't be available in a real
  deployment scenario.
- **SMOTE placement**: applied after the train/test split, and only to
  the training fold, so the test set reflects the real (imbalanced)
  failure rate — this keeps evaluation honest.
- **Scaler**: fit on the SMOTE-resampled training data; only Logistic
  Regression uses scaled input (tree models don't need it).
- **Model selection**: best model chosen by ROC-AUC on the untouched
  test set, since failure is rare and accuracy alone would be misleading.
