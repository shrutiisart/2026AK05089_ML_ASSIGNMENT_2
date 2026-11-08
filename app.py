import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🤖 Model Selection")

model_map = {
    "Logistic Regression": "models/logistic.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "SVM": "models/svm.pkl"
}

selected_model = st.sidebar.selectbox(
    "Choose Classification Model",
    list(model_map.keys())
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload Test Dataset (CSV)",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📚 Dataset Information")

st.sidebar.info("""
**Dataset:** Telco Customer Churn Dataset

**Records:** 7043

**Features:** 19

**Target Variable:** Churn

**Classes:**
- Yes = Customer Leaves
- No = Customer Stays
""")

# =====================================
# HEADER
# =====================================

st.title("📞 Telco Customer Churn Prediction Dashboard")

st.markdown("""
### Predict whether a customer will leave the telecom company (**Churn = Yes**) or stay (**Churn = No**)
""")

st.info("""
🎯 **Project Objective**

Customer churn prediction helps telecom companies identify customers who are likely to discontinue their services.

This application evaluates five machine learning models:

✅ Logistic Regression

✅ Decision Tree

✅ K-Nearest Neighbors (KNN)

✅ Naive Bayes

✅ Support Vector Machine (SVM)
""")

st.markdown("---")

# =====================================
# MODEL EVALUATION
# =====================================

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    y_true = df["Churn"]

    X = df.drop(
        "Churn",
        axis=1
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    model = joblib.load(
        model_map[selected_model]
    )

    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)

    y_prob = model.predict_proba(
        X_scaled
    )[:, 1]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    st.subheader(
        "📊 Model Performance"
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

    col2.metric(
        "AUC",
        f"{auc:.3f}"
    )

    col3.metric(
        "Precision",
        f"{precision:.3f}"
    )

    col4.metric(
        "Recall",
        f"{recall:.3f}"
    )

    col5.metric(
        "F1 Score",
        f"{f1:.3f}"
    )

    col6.metric(
        "MCC",
        f"{mcc:.3f}"
    )

    st.markdown("---")

    st.subheader(
        "📉 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        f"{selected_model} - Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    st.pyplot(fig)

else:

    st.info(
        "⬅️ Upload test_data.csv from the sidebar to evaluate a model."
    )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "BITS ML Assignment 2 | Telco Customer Churn Prediction using Machine Learning"
)