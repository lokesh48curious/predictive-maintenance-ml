"""
Predictive Maintenance — Streamlit Interface
=============================================
Loads the best model saved by train_models.py and lets a user enter
machine sensor readings to get a real-time failure prediction.

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models")

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🛠️",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "model_meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta


def build_input_row(user_inputs: dict, feature_names: list) -> pd.DataFrame:
    """Turn raw form inputs into a one-hot-encoded row matching training columns."""
    row = {col: 0 for col in feature_names}

    row["Air temperature K"] = user_inputs["air_temp"]
    row["Process temperature K"] = user_inputs["process_temp"]
    row["Rotational speed rpm"] = user_inputs["rpm"]
    row["Torque Nm"] = user_inputs["torque"]
    row["Tool wear min"] = user_inputs["tool_wear"]

    # One-hot Type column: baseline (dropped by get_dummies) is 'H' unless
    # a Type_L / Type_M column exists and matches
    type_col = f"Type_{user_inputs['machine_type']}"
    if type_col in row:
        row[type_col] = 1

    return pd.DataFrame([row])[feature_names]


def main():
    st.title("🛠️ Predictive Maintenance")
    st.caption("AI4I 2020 dataset — predicts machine failure from live sensor readings")

    try:
        model, scaler, meta = load_artifacts()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python train_models.py` first "
            "to train and save a model."
        )
        return

    st.sidebar.header("Model Info")
    st.sidebar.write(f"**Best model:** {meta['best_model_name']}")
    st.sidebar.write("**Test metrics:**")
    for k, v in meta["metrics"].items():
        st.sidebar.write(f"- {k}: {v}")

    st.subheader("Enter Machine Readings")

    col1, col2 = st.columns(2)
    with col1:
        machine_type = st.selectbox("Product Type", ["L", "M", "H"])
        air_temp = st.number_input("Air temperature [K]", value=300.0, step=0.1)
        process_temp = st.number_input("Process temperature [K]", value=310.0, step=0.1)
    with col2:
        rpm = st.number_input("Rotational speed [rpm]", value=1500, step=10)
        torque = st.number_input("Torque [Nm]", value=40.0, step=0.5)
        tool_wear = st.number_input("Tool wear [min]", value=100, step=1)

    if st.button("Predict", type="primary"):
        user_inputs = {
            "machine_type": machine_type,
            "air_temp": air_temp,
            "process_temp": process_temp,
            "rpm": rpm,
            "torque": torque,
            "tool_wear": tool_wear,
        }

        X_input = build_input_row(user_inputs, meta["feature_names"])

        if meta["uses_scaled_input"]:
            X_input_final = scaler.transform(X_input)
        else:
            X_input_final = X_input

        pred = model.predict(X_input_final)[0]
        proba = model.predict_proba(X_input_final)[0, 1]

        st.divider()
        if pred == 1:
            st.error(f"⚠️ Failure predicted — probability: {proba:.1%}")
        else:
            st.success(f"✅ No failure predicted — failure probability: {proba:.1%}")

        st.progress(min(float(proba), 1.0))


if __name__ == "__main__":
    main()
