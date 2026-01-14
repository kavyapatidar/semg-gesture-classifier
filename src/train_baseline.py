import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_baseline_classifier(X, y):
    """
    Train a baseline Logistic Regression classifier on EMG windows.
    """

    # Flatten EMG windows: (N, 102, 8) -> (N, 816)
    X_flat = X.reshape(X.shape[0], -1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression (version-safe)
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)

    print("\nBaseline Results")
    print("----------------")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model
