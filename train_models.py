# train_models.py

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocess import load_and_preprocess
from evaluate import evaluate_model


# Create model folder if not exists
if not os.path.exists("model"):
    os.makedirs("model")


# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess("bank.csv")

results = []
feature_columns = X_train.columns.tolist()

import json
with open("model/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)


# 1️⃣ Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]
results.append(("Logistic Regression", *evaluate_model(
    "Logistic Regression", y_test, y_pred, y_prob)))
joblib.dump(lr, "model/logistic.pkl")


# 2️⃣ Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]
results.append(("Decision Tree", *evaluate_model(
    "Decision Tree", y_test, y_pred, y_prob)))
joblib.dump(dt, "model/decision_tree.pkl")


# 3️⃣ KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]
results.append(("KNN", *evaluate_model(
    "KNN", y_test, y_pred, y_prob)))
joblib.dump(knn, "model/knn.pkl")


# 4️⃣ Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:, 1]
results.append(("Naive Bayes", *evaluate_model(
    "Naive Bayes", y_test, y_pred, y_prob)))
joblib.dump(nb, "model/naive_bayes.pkl")


# 5️⃣ Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
results.append(("Random Forest", *evaluate_model(
    "Random Forest", y_test, y_pred, y_prob)))
joblib.dump(rf, "model/random_forest.pkl")


# 6️⃣ XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]
results.append(("XGBoost", *evaluate_model(
    "XGBoost", y_test, y_pred, y_prob)))
joblib.dump(xgb, "model/xgboost.pkl")


# Save comparison table
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"
])

results_df.to_csv("model/model_comparison.csv", index=False)

print("\n✅ All models trained and saved successfully!")
print("📁 Models saved inside 'model/' folder")
