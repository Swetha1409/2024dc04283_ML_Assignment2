# 📊 Machine Learning Classification Comparison Web App

## 📌 Project Overview

This project implements and compares multiple Machine Learning classification models using a real-world dataset. The models are trained, evaluated using multiple performance metrics, and deployed using Streamlit for interactive testing.

The application allows users to:

- Upload a CSV dataset
- Select a machine learning model
- Evaluate performance
- View evaluation metrics
- Visualize confusion matrix

This project demonstrates an end-to-end Machine Learning workflow including:

- Data preprocessing
- Model training
- Model evaluation
- Model serialization
- Web deployment using Streamlit

---

## 🎯 Problem Statement

The objective of this project is to build and compare multiple classification algorithms and deploy them as a web application for interactive evaluation.

The models are evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC Score
- Matthews Correlation Coefficient (MCC)

---

## 📂 Dataset Description

Dataset Used: **Bank Marketing Dataset**

The dataset contains customer information used to predict whether a customer subscribes to a term deposit.

### Dataset Information

- Total Records: 4521
- Total Features: 16
- Target Variable: `y` (Binary Classification)

### Example Features

- Age
- Job
- Marital Status
- Education
- Balance
- Housing Loan
- Personal Loan
- Campaign Calls
- Previous Outcome

Target Variable:

y → 0 (No), 1 (Yes)


---

## 🤖 Machine Learning Models Implemented

The following classification algorithms were trained:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC Score
- Matthews Correlation Coefficient
- Confusion Matrix

---

## 📁 Project Structure

project-folder/
│-- app.py
│-- train_models.py
│-- preprocess.py
│-- evaluate.py
│-- requirements.txt
│-- README.md
│-- model/
│     -- logistic.pkl
│     -- decision_tree.pkl
│     -- knn.pkl
│     -- naive_bayes.pkl
│     -- random_forest.pkl
│     -- xgboost.pkl
│     -- model_comparison.csv

