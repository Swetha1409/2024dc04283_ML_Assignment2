# app.py

import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="ML Classification Comparison App",
    layout="wide"
)

st.title("📊 Machine Learning Classification Comparison")
st.write("Upload a CSV file and select a model to evaluate performance.")

# ---------------------------
# Model Dictionary
# ---------------------------
MODEL_PATHS = {
    "Logistic Regression": "model/logistic.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

# ---------------------------
# File Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Try with ; separator (for bank dataset)
        data = pd.read_csv(uploaded_file, sep=";")
    except:
        data = pd.read_csv(uploaded_file)

    st.subheader("📂 Uploaded Dataset Preview")
    st.dataframe(data.head())

    if "y" not in data.columns:
        st.error("Target column 'y' not found in dataset.")
    else:
        # Split features & target
        X = data.drop("y", axis=1)
        y = data["y"]

        # ---------------------------
        # Model Selection Dropdown
        # ---------------------------
        st.subheader("Select Model for Evaluation")
        model_choice = st.selectbox("Choose Model", list(MODEL_PATHS.keys()))

        if st.button("Evaluate Model"):
            model_path = MODEL_PATHS[model_choice]

            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
            else:
                model = joblib.load(model_path)

                # Predictions
                y_pred = model.predict(X)

                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X)[:, 1]
                else:
                    y_prob = y_pred

                # ---------------------------
                # Metrics Calculation
                # ---------------------------
                accuracy = accuracy_score(y, y_pred)
                precision = precision_score(y, y_pred)
                recall = recall_score(y, y_pred)
                f1 = f1_score(y, y_pred)
                auc = roc_auc_score(y, y_prob)
                mcc = matthews_corrcoef(y, y_pred)
                
model_choice = st.selectbox("Choose Model", list(MODEL_PATHS.keys()))

                # ---------------------------
                # Display Metrics
                # ---------------------------
                st.subheader("📈 Evaluation Metrics")
                col1, col2, col3 = st.columns(3)

                col1.metric("Accuracy", f"{accuracy:.4f}")
                col1.metric("Precision", f"{precision:.4f}")

                col2.metric("Recall", f"{recall:.4f}")
                col2.metric("F1 Score", f"{f1:.4f}")

                col3.metric("AUC Score", f"{auc:.4f}")
                col3.metric("MCC", f"{mcc:.4f}")

                # ---------------------------
                # Confusion Matrix
                # ---------------------------
                st.subheader("🔎 Confusion Matrix")
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig)
else:
    st.info("Please upload a CSV file to begin.")
