# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(path):
    # Load dataset
    data = pd.read_csv(path)

    # Encode categorical columns
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])

    # Split features and target
    X = data.drop("y", axis=1)
    y = data["y"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
