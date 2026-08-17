"""
AI4I 2020 Predictive Maintenance — Model Training Pipeline
============================================================
Loads the AI4I 2020 Predictive Maintenance dataset, cleans it, encodes
categoricals, splits into train/test, balances the training set with
SMOTE, trains Logistic Regression / Random Forest / XGBoost, evaluates
each with confusion matrices + precision/recall/F1/ROC-AUC, plots
feature importance (native + SHAP), and saves the best model to disk
for the Streamlit app to consume.

Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository)
    https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
Download `ai4i2020.csv` and place it in `data/ai4i2020.csv` before running.

Usage:
    python train_models.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_PATH = Path("data/ai4i2020.csv")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")
RANDOM_STATE = 42
TARGET_COL = "Machine failure"

# Columns that are identifiers / leak info about *which* failure occurred
# (TWF, HDF, PWF, OSF, RNF are failure-mode sub-labels — dropping them
# prevents target leakage since they perfectly encode the label).
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# 1. Load and inspect
# --------------------------------------------------------------------------
def load_and_inspect(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download 'ai4i2020.csv' from the UCI "
            "repository (AI4I 2020 Predictive Maintenance Dataset) and "
            "place it at this path."
        )

    df = pd.read_csv(path)

    print("=" * 60)
    print("DATA INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nTarget distribution:\n{df[TARGET_COL].value_counts()}")
    print(f"Failure rate: {df[TARGET_COL].mean():.4f}")
    print(f"\nSummary stats:\n{df.describe()}")

    return df


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"\nDropped {before - len(df)} duplicate rows")

    # Drop ID columns and failure-mode sub-labels (target leakage)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Handle missing values (median for numeric, mode for categorical)
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Clean up column names: XGBoost rejects names containing [, ], or <
    # (e.g. "Air temperature [K]"), so strip those characters out.
    df.columns = [
        c.strip().replace("[", "").replace("]", "").replace("<", "").strip()
        for c in df.columns
    ]

    return df


# --------------------------------------------------------------------------
# 3. One-hot encode
# --------------------------------------------------------------------------
def encode_features(df: pd.DataFrame):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]

    print(f"\nOne-hot encoding categorical columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df


# --------------------------------------------------------------------------
# 4. Split
# --------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train failure rate: {y_train.mean():.4f}")
    print(f"Test failure rate: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# 5. SMOTE (training data only)
# --------------------------------------------------------------------------
def apply_smote(X_train, y_train):
    print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# --------------------------------------------------------------------------
# 6. Train models
# --------------------------------------------------------------------------
def train_models(X_train, y_train, scaler):
    X_train_scaled = scaler.transform(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    trained = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        trained[name] = model

    return trained


# --------------------------------------------------------------------------
# 7. Evaluate: confusion matrices + precision/recall/F1/ROC-AUC
# --------------------------------------------------------------------------
def evaluate_models(trained_models, X_test, y_test, scaler):
    results = {}
    X_test_scaled = scaler.transform(X_test)

    fig, axes = plt.subplots(1, len(trained_models), figsize=(15, 4))

    for idx, (name, model) in enumerate(trained_models.items()):
        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        }

        print(f"\n{name}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Failure", "Failure"])
        disp.plot(ax=axes[idx], colorbar=False, cmap="Blues")
        axes[idx].set_title(name)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close()
    print(f"\nSaved confusion matrices -> {OUTPUTS_DIR / 'confusion_matrices.png'}")

    # ROC curves, all models overlaid
    plt.figure(figsize=(6, 5))
    for name, model in trained_models.items():
        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        y_proba = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "roc_curves.png", dpi=150)
    plt.close()
    print(f"Saved ROC curves -> {OUTPUTS_DIR / 'roc_curves.png'}")

    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# --------------------------------------------------------------------------
# 8. Feature importance (native + SHAP)
# --------------------------------------------------------------------------
def plot_feature_importance(trained_models, X_train, feature_names):
    # Native importances: RF (impurity-based) and XGBoost (gain-based)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
        model = trained_models[name]
        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=True).tail(10)
        importances.plot(kind="barh", ax=ax, color="#2b6cb0")
        ax.set_title(f"{name} — Feature Importance")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved feature importance -> {OUTPUTS_DIR / 'feature_importance.png'}")

    # SHAP summary plot for XGBoost (best model class for TreeExplainer)
    try:
        import shap

        explainer = shap.TreeExplainer(trained_models["XGBoost"])
        # Use a sample for speed on larger datasets
        sample = X_train.sample(min(1000, len(X_train)), random_state=RANDOM_STATE)
        shap_values = explainer.shap_values(sample)

        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved SHAP summary -> {OUTPUTS_DIR / 'shap_summary.png'}")
    except ImportError:
        print("shap not installed — skipping SHAP plot (pip install shap)")


# --------------------------------------------------------------------------
# 9. Save best model
# --------------------------------------------------------------------------
def save_best_model(trained_models, results, scaler, feature_names):
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]

    print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']})")

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    meta = {
        "best_model_name": best_name,
        "uses_scaled_input": best_name == "Logistic Regression",
        "feature_names": list(feature_names),
        "metrics": results[best_name],
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model -> {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved metadata -> {MODELS_DIR / 'model_meta.json'}")

    return best_name


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    df = load_and_inspect(DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    scaler = StandardScaler()
    scaler.fit(X_train_res)  # fit only on (resampled) training data

    trained_models = train_models(X_train_res, y_train_res, scaler)
    results = evaluate_models(trained_models, X_test, y_test, scaler)
    plot_feature_importance(trained_models, X_train_res, X_train.columns)
    best_name = save_best_model(trained_models, results, scaler, X_train.columns)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"All outputs saved to: {OUTPUTS_DIR}/")
    print(f"Model artifacts saved to: {MODELS_DIR}/")
    print("\nRun `streamlit run app.py` to launch the prediction interface.")


if __name__ == "__main__":
    main()
