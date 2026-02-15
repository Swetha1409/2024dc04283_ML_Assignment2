# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

# Title
st.title("Bank Term Deposit Prediction ML App")

# Sidebar - File upload
st.sidebar.header("Upload your CSV Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Sidebar - Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ("Logistic Regression",
     "Decision Tree",
     "KNN",
     "Naive Bayes",
     "Random Forest",
     "XGBoost")
)

# Function to load model
@st.cache_resource
def load_model(model_name):
    model_path = f"model/{model_name.lower().replace(' ', '_')}.pkl"
    model = joblib.load(model_path)
    return model

# Function to evaluate predictions
def evaluate(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, precision, recall, f1, auc, mcc

# Load model
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    if 'y' not in data.columns:
        st.warning("CSV must contain 'y' column as target.")
    else:
        X_test = data.drop("y", axis=1)
        y_test = data["y"]

        model = load_model(model_choice)
        y_pred = model.predict(X_test)

        # Some models might not have predict_proba
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except:
            y_prob = y_pred

        accuracy, precision, recall, f1, auc, mcc = evaluate(y_test, y_pred, y_prob)

        # Display metrics
        st.subheader("Evaluation Metrics")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**AUC Score:** {auc:.4f}")
        st.write(f"**MCC Score:** {mcc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to proceed.")
